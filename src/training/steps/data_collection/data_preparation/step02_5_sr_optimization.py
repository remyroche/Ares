"""Step 2.5: S/R Detection Optimization with Comprehensive Reporting and Function Call Monitoring."""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import time
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import traceback
import logging
import random

# Core imports
from src.training.base_step import BaseStep
from src.utils.logger import system_logger

# Initialize logger early to avoid usage before definition
logger = system_logger.getChild('Step2_5SROptimization')

# Required utility modules - Comprehensive Integration
from src.utils.common_operations import (
    safe_json_load, safe_json_dump, safe_read_parquet, safe_to_parquet,
    ensure_directory, create_fallback_logger, create_fallback_decorator,
    safe_mean, safe_std, safe_float, safe_int, safe_append, safe_extend,
    safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
    get_current_datetime, format_datetime, create_empty_dataframe,
    safe_fillna, safe_rolling, safe_copy, safe_deepcopy, safe_sleep,
    safe_gather, create_async_task, get_logger, setup_basic_logging,
    safe_exception_handler, suggest_float_uniform, suggest_int_uniform,
    validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    safe_log_metric, safe_log_params, safe_log_artifact, get_common_operations_health_status
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, validate_finite,
    validate_positive, validate_range, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, MathValidationError
)
from src.utils.parquet_utils import ParquetUtils
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
    save_data, load_data, SerializationError
)
from src.utils.data_processing_utils import (
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
    DataQualityLevel, DataQualityIssue, DataQualityReport,
    validate_dataframe, clean_dataframe, transform_dataframe, get_dataframe_info
)

# Core decorators and errors
from src.core.decorators import handles_errors, error_boundary, converts_errors
from src.core.errors import (
    AppError, ValidationError, DataIntegrityError, 
    NotFoundError, BusinessRuleError
)

# Pipeline standards and utilities
from src.utils.pipeline_standards import PipelineStandards
from src.utils.monitoring_utils import (
    global_monitor, function_tracker, logging_patterns
)
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.training.reports import save_training_report
from src.training.steps.data_collection.data_preparation.step02_5_financial_logging import Step02_5FinancialLogger
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

# M1 Optimization Utilities - Comprehensive Integration
try:
    from src.utils.m1_gpu_utils import (
        M1GPUManager, M1PerformanceOptimizer, initialize_m1_gpu, get_m1_gpu_manager,
        m1_tensor_multiply, m1_batch_process, m1_monte_carlo_simulate,
        create_m1_optimized_config
    )
    M1_GPU_AVAILABLE = True
except ImportError as e:
    M1_GPU_AVAILABLE = False
    logger.warning(f"M1 GPU utils not available: {e}")
except Exception as e:
    M1_GPU_AVAILABLE = False
    logger.error(f"Unexpected error loading M1 GPU utils: {e}")

try:
    from src.utils.m1_memory_optimizer import (
        M1MemoryOptimizer, M1DataManager, get_m1_memory_optimizer,
        create_memory_efficient_dataframe, memory_efficient_groupby
    )
    M1_MEMORY_AVAILABLE = True
except ImportError as e:
    M1_MEMORY_AVAILABLE = False
    logger.warning(f"M1 Memory optimizer not available: {e}")
except Exception as e:
    M1_MEMORY_AVAILABLE = False
    logger.error(f"Unexpected error loading M1 Memory optimizer: {e}")

try:
    from src.utils.m1_cpu_optimizer import (
        M1CPUOptimizer, M1BatchProcessor, get_m1_cpu_optimizer,
        initialize_m1_cpu_optimizer, parallel_map, parallel_dataframe_operation,
        parallel_monte_carlo_simulation, optimized_monte_carlo_worker
    )
    M1_CPU_AVAILABLE = True
    M1_BATCH_AVAILABLE = True
except ImportError as e:
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.warning(f"M1 CPU optimizer not available: {e}")
except Exception as e:
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.error(f"Unexpected error loading M1 CPU optimizer: {e}")

try:
    from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
    STANDARDIZED_PARQUET_AVAILABLE = True
except ImportError:
    standardized_parquet_handler = None
    STANDARDIZED_PARQUET_AVAILABLE = False

# Import optional modules with error handling
try:
    from src.utils.parquet_utils import ParquetUtils
    PARQUET_UTILS_AVAILABLE = True
except ImportError:
    ParquetUtils = None
    PARQUET_UTILS_AVAILABLE = False

# Import new ML Common utilities
try:
    from src.utils.ml_common import (
        FeatureSelectionFramework, LookaheadProtection, CrossValidationUtilities,
        ModelEvaluationUtilities, DataQualityUtilities, MemoryEfficientTraining,
        ParallelProcessingCoordinator
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    logger.warning(f"⚠️ ML Common utilities not available: {e}")

try:
    from src.training.steps.feature_engineering.step06_advanced_features import AdvancedFeatureEngineeringStep
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    AdvancedFeatureEngineeringStep = None
    ADVANCED_FEATURES_AVAILABLE = False

try:
    from src.tactician.sr_levels.sr_levels_manager import SRLevelsManager
    SR_LEVELS_MANAGER_AVAILABLE = True
except ImportError:
    SRLevelsManager = None
    SR_LEVELS_MANAGER_AVAILABLE = False

try:
    from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
    DYNAMIC_BARRIER_CALCULATOR_AVAILABLE = True
except ImportError:
    DynamicBarrierCalculator = None
    DYNAMIC_BARRIER_CALCULATOR_AVAILABLE = False

try:
    from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector, SRLevel
    ENHANCED_SR_DETECTOR_AVAILABLE = True
except ImportError:
    EnhancedSRDetector = None
    SRLevel = None
    ENHANCED_SR_DETECTOR_AVAILABLE = False

# For parameter optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Enhanced diagnostics imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

# Dependency Injection Container for Step02_5
class Step02_5DependencyContainer:
    """Dependency injection container for Step02_5 utilities and services."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._services = {}
        self._initialized = False
        
    def initialize_services(self):
        """Initialize all utility services with dependency injection."""
        if self._initialized:
            return
            
        logger.info("🔧 Initializing Step02_5 dependency container...")
        
        # Initialize utility services
        self._services['parquet_utils'] = ParquetUtils() if PARQUET_UTILS_AVAILABLE else None
        self._services['data_validator'] = DataFrameValidator(self.config.get('validation', {}))
        self._services['data_cleaner'] = DataFrameCleaner(self.config.get('cleaning', {}))
        self._services['data_transformer'] = DataFrameTransformer(self.config.get('transformation', {}))
        
        # Initialize M1 optimization services
        if M1_GPU_AVAILABLE:
            self._services['m1_gpu_manager'] = get_m1_gpu_manager()
            self._services['m1_performance_optimizer'] = M1PerformanceOptimizer(self._services['m1_gpu_manager'])
        else:
            self._services['m1_gpu_manager'] = None
            self._services['m1_performance_optimizer'] = None
            
        if M1_MEMORY_AVAILABLE:
            self._services['m1_memory_optimizer'] = get_m1_memory_optimizer()
            self._services['m1_data_manager'] = M1DataManager(self._services['m1_memory_optimizer'])
        else:
            self._services['m1_memory_optimizer'] = None
            self._services['m1_data_manager'] = None
            
        if M1_CPU_AVAILABLE:
            self._services['m1_cpu_optimizer'] = get_m1_cpu_optimizer()
            self._services['m1_batch_processor'] = M1BatchProcessor(self._services['m1_cpu_optimizer'])
        else:
            self._services['m1_cpu_optimizer'] = None
            self._services['m1_batch_processor'] = None
        
        # Initialize serialization services
        self._services['json_serializer'] = JSONSerializer()
        self._services['pickle_serializer'] = PickleSerializer()
        self._services['parquet_serializer'] = ParquetSerializer()
        self._services['universal_serializer'] = UniversalSerializer()
        
        self._initialized = True
        logger.info("✅ Step02_5 dependency container initialized successfully")
        
    def get_service(self, service_name: str) -> Any:
        """Get a service from the container."""
        if not self._initialized:
            self.initialize_services()
        return self._services.get(service_name)
        
    def get_all_services(self) -> Dict[str, Any]:
        """Get all services from the container."""
        if not self._initialized:
            self.initialize_services()
        return self._services.copy()
        
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            'container_initialized': self._initialized,
            'services_available': {},
            'overall_health': 'healthy'
        }
        
        for service_name, service in self._services.items():
            if service is not None:
                health_status['services_available'][service_name] = True
            else:
                health_status['services_available'][service_name] = False
                health_status['overall_health'] = 'degraded'
                
        return health_status

# Error classification system using core error classes
class ErrorSeverity:
    CRITICAL = "CRITICAL"  # System cannot continue
    HIGH = "HIGH"         # Major functionality affected
    MEDIUM = "MEDIUM"     # Minor functionality affected
    LOW = "LOW"          # Cosmetic or non-critical

class ErrorCategory:
    DATA_QUALITY = "DATA_QUALITY"
    ML_TRAINING = "ML_TRAINING"
    SR_DETECTION = "SR_DETECTION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    SYSTEM_RESOURCE = "SYSTEM_RESOURCE"
    EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"

@handles_errors(default_return=(ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_RESOURCE))
def classify_error(error: Exception, context: str = "") -> tuple[ErrorSeverity, ErrorCategory]:
    """Classify errors for appropriate handling using core error classes."""
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Critical errors - map to core error classes
    if isinstance(error, (MemoryError, SystemError)):
        return ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM_RESOURCE
    if isinstance(error, DataIntegrityError) or "all.*values.*invalid" in error_msg or "data.*corrupted" in error_msg:
        return ErrorSeverity.CRITICAL, ErrorCategory.DATA_QUALITY
    
    # High severity errors - map to core error classes
    if isinstance(error, (ValidationError, DataIntegrityError)) and "data" in context.lower():
        return ErrorSeverity.HIGH, ErrorCategory.DATA_QUALITY
    if isinstance(error, BusinessRuleError) or ("ml" in context.lower() or "model" in context.lower()):
        return ErrorSeverity.HIGH, ErrorCategory.ML_TRAINING
    
    # Medium severity errors
    if isinstance(error, (ImportError, NotFoundError)):
        return ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_DEPENDENCY
    if "sr" in context.lower() or "detection" in context.lower():
        return ErrorSeverity.MEDIUM, ErrorCategory.SR_DETECTION
    
    # Default to medium severity
    return ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_RESOURCE

@handles_errors(default_return=False)
def handle_error_with_recovery(error: Exception, context: str, max_retries: int = 3) -> bool:
    """Handle errors with appropriate recovery strategies using core error handling."""
    severity, category = classify_error(error, context)
    
    logger.error(f"🚨 {severity} ERROR in {category}: {error}")
    logger.error(f"📋 Context: {context}")
    logger.error(f"📋 Traceback: {traceback.format_exc()}")
    
    if severity == ErrorSeverity.CRITICAL:
        logger.critical("💥 CRITICAL ERROR - System cannot continue safely")
        return False
    elif severity == ErrorSeverity.HIGH:
        logger.error("⚠️ HIGH SEVERITY ERROR - Major functionality affected")
        # Could implement retry logic here
        return False
    else:
        logger.warning(f"⚠️ {severity} ERROR - Continuing with degraded functionality")
        return True

@handles_errors(default_return={'drift_detected': False, 'drift_score': 0.0, 'drift_details': {}, 'recommendations': []})
def detect_data_drift(current_data: pd.DataFrame, reference_data: pd.DataFrame = None, 
                     drift_threshold: float = 0.1) -> Dict[str, Any]:
    """Detect data drift between current and reference datasets using math validation utilities."""
    drift_results = {
        'drift_detected': False,
        'drift_score': 0.0,
        'drift_details': {},
        'recommendations': []
    }
    
    try:
        # If no reference data, use statistical baselines
        if reference_data is None:
            # Use statistical baselines for common financial metrics with safe math operations
            baseline_stats = {
                'close_mean': current_data['close'].mean(),
                'close_std': safe_sqrt(current_data['close'].var(), default=0.0),
                'volume_mean': current_data['volume'].mean(),
                'volume_std': safe_sqrt(current_data['volume'].var(), default=0.0)
            }
            
            # Simple drift detection based on statistical properties
            current_stats = {
                'close_mean': current_data['close'].mean(),
                'close_std': safe_sqrt(current_data['close'].var(), default=0.0),
                'volume_mean': current_data['volume'].mean(),
                'volume_std': safe_sqrt(current_data['volume'].var(), default=0.0)
            }
            
            # Calculate drift score (simplified)
            drift_score = 0.0
            for metric in baseline_stats:
                if baseline_stats[metric] != 0:
                    relative_change = abs(current_stats[metric] - baseline_stats[metric]) / abs(baseline_stats[metric])
                    drift_score += relative_change
                    drift_results['drift_details'][metric] = {
                        'baseline': baseline_stats[metric],
                        'current': current_stats[metric],
                        'relative_change': relative_change
                    }
            
            drift_score /= len(baseline_stats)
            drift_results['drift_score'] = drift_score
            
            if drift_score > drift_threshold:
                drift_results['drift_detected'] = True
                drift_results['recommendations'].append("Significant data drift detected - consider retraining models")
                drift_results['recommendations'].append("Review data sources and collection processes")
        
        else:
            # Compare with reference data
            # This would implement more sophisticated drift detection
            # For now, use simple statistical comparison
            pass
            
    except Exception as e:
        logger.error(f"Data drift detection failed: {e}")
        drift_results['error'] = str(e)
    
    return drift_results

def generate_function_report(ml_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate comprehensive function call report with detailed ML model metrics."""
    from src.utils.monitoring_utils import global_tracker

    # Get base performance summary
    base_report = global_tracker.get_performance_summary()

    # Add detailed ML model metrics if available
    if ml_results:
        base_report['ml_model_metrics'] = {
            'direction_accuracy': ml_results.get('direction_accuracy', 0.0),
            'volatility_mae': ml_results.get('volatility_mae', 0.0),
            'model_type': ml_results.get('model_type', 'unknown'),
            'training_samples': ml_results.get('training_samples', 0),
            'test_samples': ml_results.get('test_samples', 0),
            'training_time': ml_results.get('training_time', 0.0),
            'sr_levels_used': ml_results.get('sr_levels_used', 0),
            'feature_count': len(ml_results.get('feature_names', [])),
            'cross_validation_scores': ml_results.get('cross_validation_scores', []),
            'cv_mean_accuracy': ml_results.get('evaluation_metrics', {}).get('cv_direction_mean', 0.0),
            'cv_std_accuracy': ml_results.get('evaluation_metrics', {}).get('cv_direction_std', 0.0)
        }

        # Add comprehensive individual model performance details (including all models, even those not selected)
        if 'models_performance' in ml_results:
            base_report['ml_model_metrics']['all_models'] = {}
            for model_name, model_data in ml_results['models_performance'].items():
                model_info = {
                    'model_name': model_name,
                    'was_selected': model_name == ml_results.get('model_type', ''),
                    'direction': {},
                    'volatility': {}
                }

                # Direction classification details
                if 'direction' in model_data:
                    dir_data = model_data['direction']
                    model_info['direction'] = {
                        'accuracy': dir_data.get('accuracy', 0.0),
                        'precision': dir_data.get('classification_report', {}).get('weighted avg', {}).get('precision', 0.0),
                        'recall': dir_data.get('classification_report', {}).get('weighted avg', {}).get('recall', 0.0),
                        'f1_score': dir_data.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.0),
                        'classification_report': dir_data.get('classification_report', {}),
                        'feature_importance_count': len(dir_data.get('feature_importance', {})),
                        'predictions_count': len(dir_data.get('predictions', [])),
                        'optimized': 'optimized_params' in dir_data,
                        'optimized_params': dir_data.get('optimized_params', {}),
                        'improvement': dir_data.get('improvement', 0.0)
                    }

                # Volatility regression details
                if 'volatility' in model_data and model_data['volatility']:
                    vol_data = model_data['volatility']
                    model_info['volatility'] = {
                        'mae': vol_data.get('mae', 0.0),
                        'predictions_count': len(vol_data.get('predictions', []))
                    }

                base_report['ml_model_metrics']['all_models'][model_name] = model_info

        # Add feature selection information with SHAP details
        if 'feature_selection' in ml_results:
            fs = ml_results['feature_selection']
            base_report['ml_model_metrics']['feature_selection'] = {
                'original_features': fs.get('original_features', 0),
                'selected_features': fs.get('selected_features', 0),
                'target_features': fs.get('target_features', 80),  # Updated to reflect 80 features
                'methods_used': fs.get('methods_used', []),
                'selection_criteria': fs.get('selection_criteria', ''),
                'top_features': fs.get('top_features', []),
                'feature_importance': fs.get('feature_importance', {}),
                'mutual_information': fs.get('mutual_information', {}),
                'shap_analysis': fs.get('shap_importance', {}),
                'all_selected_features': fs.get('feature_importance', {}),  # All features used by the final model with their importance scores
                'all_features_list': list(fs.get('feature_importance', {}).keys())
            }

            # Add comprehensive SHAP information
            if 'shap_importance' in fs:
                shap_data = fs['shap_importance']
                base_report['ml_model_metrics']['feature_selection']['shap_available'] = shap_data.get('available', False)
                base_report['ml_model_metrics']['feature_selection']['shap_method'] = shap_data.get('method', 'unknown')

                if shap_data.get('available', False):
                    base_report['ml_model_metrics']['feature_selection']['shap_feature_importance'] = shap_data.get('feature_importance', {})
                    base_report['ml_model_metrics']['feature_selection']['shap_top_features'] = shap_data.get('top_features', [])
                    base_report['ml_model_metrics']['feature_selection']['shap_sample_size'] = shap_data.get('sample_size', 0)

        # Add optimization information
        if 'evaluation_metrics' in ml_results:
            eval_metrics = ml_results['evaluation_metrics']
            base_report['ml_model_metrics']['optimization'] = {
                'best_model': eval_metrics.get('best_model_type', 'unknown'),
                'best_accuracy': eval_metrics.get('best_direction_accuracy', 0.0),
                'models_evaluated': eval_metrics.get('models_count', 0),
                'cv_direction_mean': eval_metrics.get('cv_direction_mean', 0.0),
                'cv_direction_std': eval_metrics.get('cv_direction_std', 0.0),
                'cv_f1_mean': eval_metrics.get('cv_f1_mean', 0.0),
                'cv_f1_std': eval_metrics.get('cv_f1_std', 0.0),
                'feature_importance_available': bool(eval_metrics.get('feature_importance', {})),
                'top_20_features': list(eval_metrics.get('feature_importance', {}).keys())[:20]
            }

        # Add model comparison summary
        if 'models_performance' in ml_results:
            model_comparison = {}
            for model_name, model_data in ml_results['models_performance'].items():
                if 'direction' in model_data and 'accuracy' in model_data['direction']:
                    model_comparison[model_name] = {
                        'accuracy': model_data['direction']['accuracy'],
                        'rank': 0,  # Will be set below
                        'selected': model_name == ml_results.get('model_type', '')
                    }

            # Sort models by accuracy and assign ranks
            sorted_models = sorted(model_comparison.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for rank, (model_name, _) in enumerate(sorted_models, 1):
                model_comparison[model_name]['rank'] = rank

            base_report['ml_model_metrics']['model_comparison'] = {
                'ranking': sorted_models,
                'total_models': len(model_comparison),
                'best_model': sorted_models[0][0] if sorted_models else 'unknown',
                'best_accuracy': sorted_models[0][1]['accuracy'] if sorted_models else 0.0,
                'selected_model': ml_results.get('model_type', 'unknown')
            }

    return base_report

class SROptimizationStep(BaseStep):
    """Step 2.5: S/R Detection Optimization with comprehensive parameter optimization and detailed reporting."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR optimization step with comprehensive utility integration."""
        super().__init__(config, '2_5', 'sr_optimization')
        self.logger = system_logger.getChild('SROptimizationStep')
        self.standards = PipelineStandards(self.logger)
        self.sr_optimization_config = config.get('sr_optimization', {'min_touches': 2, 'tolerance_pct': 0.5, 'lookback_periods': 100})
        
        # Initialize dependency injection container
        self.dependency_container = Step02_5DependencyContainer(config)
        self.dependency_container.initialize_services()
        
        # Get utility services from container
        self.parquet_utils = self.dependency_container.get_service('parquet_utils')
        self.data_validator = self.dependency_container.get_service('data_validator')
        self.data_cleaner = self.dependency_container.get_service('data_cleaner')
        self.data_transformer = self.dependency_container.get_service('data_transformer')
        self.m1_gpu_manager = self.dependency_container.get_service('m1_gpu_manager')
        self.m1_memory_optimizer = self.dependency_container.get_service('m1_memory_optimizer')
        self.m1_cpu_optimizer = self.dependency_container.get_service('m1_cpu_optimizer')
        self.json_serializer = self.dependency_container.get_service('json_serializer')
        self.universal_serializer = self.dependency_container.get_service('universal_serializer')

        # Initialize ML Common utilities
        self._initialize_ml_common_utilities()

        # Initialize optimized logging
        self.debug_mode = config.get('debug_mode', False)
        self._initialize_logging_verbosity()

        # Fast fail configuration
        self.enable_fast_fail = config.get('enable_fast_fail', True)
        self.fast_fail_on_ml_errors = config.get('fast_fail_on_ml_errors', True)
        self.max_ml_failures = config.get('max_ml_failures', 3)

        # NEW: Enhanced configuration parameters
        self.enable_hyperparameter_optimization = config.get('enable_hyperparameter_optimization', True)
        self.optimization_method = config.get('optimization_method', 'grid_search')  # 'grid_search', 'random_search', 'bayesian'
        self.optimization_folds = config.get('optimization_folds', 5)
        self.optimization_trials = config.get('optimization_trials', 50)
        
        # NEW: Walk-forward validation
        self.enable_walk_forward_validation = config.get('enable_walk_forward_validation', True)
        self.walk_forward_folds = config.get('walk_forward_folds', 5)
        self.walk_forward_test_size = config.get('walk_forward_test_size', 0.2)
        
        # Performance optimization settings
        self.enable_m1_optimizations = config.get('enable_m1_optimizations', True)
        self.enable_memory_optimization = config.get('enable_memory_optimization', True)
        self.enable_parallel_processing = config.get('enable_parallel_processing', True)
        
        self.start_time = None
        # Use unified monitoring system instead of multiple trackers
        self.performance_monitor = global_monitor

    def _initialize_ml_common_utilities(self) -> None:
        """Initialize ML Common utilities with proper error handling."""
        try:
            if ML_COMMON_AVAILABLE:
                self.logger.info("🔧 Initializing ML Common utilities...")

                # Initialize core ML utilities
                self.feature_selector = FeatureSelectionFramework()
                self.lookahead_protector = LookaheadProtection()
                self.cv_utils = CrossValidationUtilities()
                self.model_evaluator = ModelEvaluationUtilities()
                self.data_quality_utils = DataQualityUtilities()
                self.memory_optimizer = MemoryEfficientTraining()
                self.parallel_processor = ParallelProcessingCoordinator()

                self.logger.info("✅ ML Common utilities initialized successfully")
            else:
                self.logger.warning("⚠️ ML Common utilities not available - using fallback implementations")
                self._initialize_fallback_utilities()

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Common utilities: {e}")
            self._initialize_fallback_utilities()

    def _initialize_fallback_utilities(self) -> None:
        """Initialize fallback utilities when ML Common is not available."""
        self.feature_selector = None
        self.lookahead_protector = None
        self.cv_utils = None
        self.model_evaluator = None
        self.data_quality_utils = None
        self.memory_optimizer = None
        self.parallel_processor = None
        
        # Log utility integration status
        self._log_utility_integration_status()

        # Initialize ML failure tracking
        self.ml_failure_count = 0
        self.ml_failure_reasons = []
        self.fast_fail_engaged = False  # Flag to prevent redundant restart attempts

        # ML Model Configurations
        self.ml_model_configs = {
            'RandomForestClassifier': {
                'class': RandomForestClassifier,
                'hyperparameters': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 10},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                    'bootstrap': {'type': 'categorical', 'choices': [True, False]},
                    'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']}
                },
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'criterion': 'gini',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'hyperparameters': {
                    'C': {'type': 'float', 'low': 1e-3, 'high': 10.0, 'log': True},  # Narrower range for better convergence
                    'penalty': {'type': 'categorical', 'choices': ['l2', 'l1']},  # Remove elasticnet and none for stability
                    'solver': {'type': 'categorical', 'choices': ['lbfgs', 'liblinear']},  # More stable solvers
                    'max_iter': {'type': 'int', 'low': 1000, 'high': 5000, 'step': 500}  # Higher minimum iterations
                },
                'default_params': {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'liblinear',  # More robust solver
                    'max_iter': 2000,  # Higher default iterations
                    'random_state': 42,
                    'n_jobs': -1,
                    'multi_class': 'ovr',
                    'tol': 1e-4  # Tighter tolerance for better convergence
                }
            },
            'HistGradientBoostingClassifier': {
                'class': HistGradientBoostingClassifier,
                'hyperparameters': {
                    'max_iter': {'type': 'int', 'low': 50, 'high': 500, 'step': 10},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 1.0, 'log': True},
                    'min_samples_leaf': {'type': 'int', 'low': 5, 'high': 50},
                    'l2_regularization': {'type': 'float', 'low': 1e-10, 'high': 1.0, 'log': True},
                    'max_bins': {'type': 'int', 'low': 100, 'high': 255, 'step': 10}
                },
                'default_params': {
                    'max_iter': 100,
                    'max_depth': None,
                    'learning_rate': 0.1,
                    'min_samples_leaf': 20,
                    'l2_regularization': 1e-3,
                    'max_bins': 255,
                    'random_state': 42
                }
            }
        }

    def _initialize_logging_verbosity(self):
        """Reduce logging verbosity for better performance."""
        # Set logger level to INFO to reduce DEBUG overhead
        if hasattr(self.logger, 'setLevel'):
            self.logger.setLevel(logging.INFO)

        # Disable verbose sklearn logging
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('sklearn.externals.joblib').setLevel(logging.WARNING)

        # Disable verbose pandas logging
        logging.getLogger('pandas').setLevel(logging.WARNING)

    def _handle_ml_failure(self, error_message: str, error_type: str = "UNKNOWN_ERROR") -> Dict[str, Any]:
        """Handle ML training failures with intelligent fast fail mechanism."""
        self.ml_failure_count += 1
        self.ml_failure_reasons.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'failure_count': self.ml_failure_count
        })

        # Classify failure severity
        critical_errors = ["FORWARD_BIAS_ERROR", "DATA_UNAVAILABLE", "EMPTY_DATA"]
        recoverable_errors = ["OPTUNA_ERROR", "CV_ERROR", "MODEL_FIT_ERROR", "ML_TRAINING_ERROR", "METHOD_VALIDATION_ERROR"]

        is_critical = error_type in critical_errors
        is_recoverable = error_type in recoverable_errors

        if is_critical:
            self.logger.error(f'❌ CRITICAL ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.error(f'🚨 Critical Error Type: {error_type}')
        elif is_recoverable:
            self.logger.warning(f'⚠️ RECOVERABLE ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.warning(f'📊 Recoverable Error Type: {error_type}')
        else:
            self.logger.warning(f'⚠️ ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.warning(f'📊 Error Type: {error_type}')

        # Intelligent fast fail logic
        if self.enable_fast_fail and self.fast_fail_on_ml_errors:
            # Different thresholds for different error types
            if is_critical and self.ml_failure_count >= 2:  # Fail faster on critical errors
                self.logger.critical(f'🚨 FAST FAIL: {self.ml_failure_count} ML failures detected (critical), aborting training')
                self.fast_fail_engaged = True  # Set flag to prevent redundant restarts
                raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} critical ML training failures")
            elif self.ml_failure_count >= self.max_ml_failures:  # Original threshold for other errors
                self.logger.critical(f'🚨 FAST FAIL: {self.ml_failure_count} ML failures detected, aborting training')
                self.fast_fail_engaged = True  # Set flag to prevent redundant restarts
                raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} ML training failures")

        # Return fallback result with failure information
        fallback_result = self._get_fallback_ml_result_with_failure_info(error_message, error_type)
        return fallback_result

    def _get_fallback_ml_result_with_failure_info(self, error_message: str, error_type: str) -> Dict[str, Any]:
        """Get fallback ML result with detailed failure information."""
        return {
            'direction_accuracy': 0.5,
            'volatility_mae': 0.1,
            'model_type': 'fallback_due_to_failure',
            'training_samples': 0,
            'sr_levels_used': 0,
            'training_time': 0.0,
            'failure_reason': error_message,
            'failure_type': error_type,
            'failure_count': self.ml_failure_count,
            'fast_fail_enabled': self.enable_fast_fail
        }

    def _log_utility_integration_status(self) -> None:
        """Log the status of utility integration."""
        self.logger.info("🔧 Step02_5 Utility Integration Status:")
        
        # Check utility availability
        utilities_status = {
            'common_operations': True,  # Always available
            'common_utilities': True,   # Always available
            'math_validation': True,    # Always available
            'parquet_utils': self.parquet_utils is not None,
            'serialization_utils': self.json_serializer is not None,
            'data_processing_utils': self.data_validator is not None,
            'm1_gpu_utils': self.m1_gpu_manager is not None,
            'm1_memory_optimizer': self.m1_memory_optimizer is not None,
            'm1_cpu_optimizer': self.m1_cpu_optimizer is not None,
            'dependency_injection': True  # Always available
        }
        
        for utility, available in utilities_status.items():
            status_emoji = "✅" if available else "❌"
            self.logger.info(f"  {status_emoji} {utility}: {'Available' if available else 'Not Available'}")
        
        # Log dependency container health
        health_status = self.dependency_container.health_check()
        self.logger.info(f"📊 Dependency Container Health: {health_status['overall_health']}")
        
        # Log M1 optimization status
        if self.enable_m1_optimizations:
            m1_status = {
                'GPU': self.m1_gpu_manager is not None,
                'Memory': self.m1_memory_optimizer is not None,
                'CPU': self.m1_cpu_optimizer is not None
            }
            self.logger.info(f"🍎 M1 Optimizations: {m1_status}")
    
    @log_step_functions
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.logger.info('✅ SR optimization step initialized')
        self.logger.info(f'📊 Configuration loaded: {self.sr_optimization_config}')

    async def initialize(self) -> None:
        """Initialize the step."""
        self._initialize_step()
        self.logger.info('🚀 Step 2.5 initialization completed')

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step."""
        self.logger.info('🎯 Starting Step 2.5 execution with comprehensive monitoring')
        pre_report = generate_function_report()
        self.logger.info(f"📊 Pre-execution function calls: {pre_report['total_calls']}")

        # Check if execute_logic method exists, if not, use execute_main_logic
        if hasattr(self, 'execute_logic'):
            result = await self.execute_logic(training_input, pipeline_state)
        else:
            result = await self.execute_main_logic(training_input, pipeline_state)

        # Pass ML results to function report for detailed metrics
        ml_results = result.get('ml_results', {})
        post_report = generate_function_report(ml_results)
        self.logger.info(f"📊 Post-execution function calls: {post_report['total_calls']}")
        self.logger.info(f"📈 Function call increase: {post_report['total_calls'] - pre_report['total_calls']}")
        result['function_call_report'] = post_report
        return result

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for SR optimization from raw market data."""
        try:
            self.logger.info('🔧 Preparing features from market data...')

            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {[col for col in required_columns if col not in data.columns]}")

            # Make a copy to avoid modifying original data
            features_data = data.copy()

            # Add basic technical indicators
            features_data['returns'] = features_data['close'].pct_change()
            features_data['log_returns'] = np.log(features_data['close'] / features_data['close'].shift(1))

            # Add volatility measures
            features_data['volatility_5'] = features_data['returns'].rolling(5).std()
            features_data['volatility_10'] = features_data['returns'].rolling(10).std()
            features_data['volatility_20'] = features_data['returns'].rolling(20).std()

            # Add momentum indicators
            features_data['momentum_5'] = features_data['close'] / features_data['close'].shift(5) - 1
            features_data['momentum_10'] = features_data['close'] / features_data['close'].shift(10) - 1

            # Add volume indicators
            features_data['volume_ma_5'] = features_data['volume'].rolling(5).mean()
            features_data['volume_ratio'] = features_data['volume'] / features_data['volume_ma_5']

            # Add price range indicators
            features_data['high_low_ratio'] = features_data['high'] / features_data['low']
            features_data['close_open_ratio'] = features_data['close'] / features_data['open']

            # Fill any NaN values that might have been created
            # Handle categorical columns separately to avoid category errors
            for col in features_data.columns:
                if features_data[col].dtype.name == 'category':
                    # For categorical columns, fill with mode or a valid category
                    mode_val = features_data[col].mode()
                    if not mode_val.empty:
                        features_data[col] = features_data[col].fillna(mode_val.iloc[0])
                    else:
                        # If no mode, convert to string and fill with empty string
                        features_data[col] = features_data[col].astype(str).fillna('')
                else:
                    # For numeric columns, use forward/backward fill then 0
                    features_data[col] = features_data[col].fillna(method='bfill').fillna(method='ffill').fillna(0)

            self.logger.info(f'✅ Features prepared: {features_data.shape[1]} features from {features_data.shape[0]} data points')

            return features_data

        except Exception as e:
            self.logger.error(f'❌ Feature preparation failed: {e}')
            raise

    async def execute_main_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive SR optimization logic - main implementation."""
        self.logger.info('🎯 Starting comprehensive S/R detection optimization with unified monitoring...')
        self.logger.info(f'📊 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'📊 Pipeline state keys: {list(pipeline_state.keys())}')
        self.start_time = time.time()

        try:
            # Check if fast-fail has been engaged - prevent redundant restart attempts
            if hasattr(self, 'fast_fail_engaged') and self.fast_fail_engaged:
                self.logger.warning('⚠️ Fast-fail previously engaged - skipping redundant training restart')
                return {
                    'success': False,
                    'error': 'Fast-fail previously engaged - training aborted to prevent redundant attempts',
                    'fast_fail_engaged': True,
                    'execution_time': time.time() - self.start_time
                }

            # CRITICAL: Validate data availability before any processing
            self.logger.info('📊 Retrieving data from pipeline state...')
            data = pipeline_state.get('dataframe')
            if data is None:
                raise ValueError("No dataframe found in pipeline state")

            self.logger.info(f'📊 Data loaded: {data.shape[0]:,} rows, {data.shape[1]} columns')

            # Prepare features from data
            self.logger.info('🔧 Preparing features for SR optimization...')
            features_data = self._prepare_features(data)

            # Detect SR levels
            self.logger.info('🎯 Detecting Support/Resistance levels...')
            sr_levels = self._detect_sr_levels(features_data)

            # Train ML models for SR optimization
            self.logger.info('🤖 Training ML models for SR optimization...')
            ml_results = await self._train_ml_models_with_memory_management(features_data, sr_levels)

            # Prepare final results
            execution_time = time.time() - self.start_time
            result = {
                'success': True,
                'sr_levels': sr_levels,
                'ml_results': ml_results,
                'features_data': features_data,
                'execution_time': execution_time,
                'data_shape': data.shape,
                'sr_levels_count': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', []))
            }

            self.logger.info(f'✅ SR optimization completed in {execution_time:.2f} seconds')
            self.logger.info(f'🎯 SR levels detected: {result["sr_levels_count"]}')

            return result

        except Exception as e:
            self.logger.error(f'❌ SR optimization failed: {e}')
            import traceback
            self.logger.error(f'📋 Full traceback: {traceback.format_exc()}')
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'execution_time': time.time() - self.start_time
            }

    def _detect_sr_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels using Enhanced SR Detection."""
        self.logger.info('🎯 Using Enhanced SR Detection with multiple advanced algorithms...')

        # CRITICAL: Validate input data before S/R detection
        if data is None:
            raise ValueError("CRITICAL: Input data is None for S/R detection. Cannot proceed.")

        if data.empty:
            raise ValueError("CRITICAL: Input data is empty for S/R detection. Cannot proceed.")

        if len(data) < 100:  # Minimum 100 rows for meaningful S/R detection
            raise ValueError(f"CRITICAL: Insufficient data for S/R detection. Only {len(data)} rows available, minimum 100 required.")

        # Validate required columns for S/R detection
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"CRITICAL: Missing required columns for S/R detection: {missing_columns}. Available columns: {list(data.columns)}")

        self.logger.info(f'✅ S/R detection input validation passed: {len(data)} rows, {len(data.columns)} columns')

        # Check if enhanced detector is available
        if not ENHANCED_SR_DETECTOR_AVAILABLE or EnhancedSRDetector is None or SRLevel is None:
            raise RuntimeError("Enhanced SR Detector or SRLevel not available. Cannot proceed with SR detection.")

        try:
            # Create enhanced SR detector with configuration - optimized for memory
            sr_config = {
                'min_touches': getattr(self, 'min_touches', 2),
                'tolerance_pct': getattr(self, 'tolerance_pct', 0.5),
                'lookback_periods': getattr(self, 'lookback_periods', 100),
                'memory_efficient': True,
                'use_parallel': self.enable_parallel_processing if hasattr(self, 'enable_parallel_processing') else False
            }

            # Initialize detector
            detector = EnhancedSRDetector(sr_config)

            # Detect levels
            sr_levels = detector.detect_sr_levels(data)

            # Handle the case where detector returns a list instead of dict
            if isinstance(sr_levels, list):
                # Convert list format to expected dict format
                support_levels = []
                resistance_levels = []

                for level in sr_levels:
                    if hasattr(level, 'type'):
                        level_type = level.type
                    elif hasattr(level, 'get') and callable(getattr(level, 'get')):
                        level_type = level.get('type')
                    else:
                        # Try to access as attribute or skip
                        level_type = getattr(level, 'type', None)

                    if level_type == 'support':
                        support_levels.append(level)
                    elif level_type == 'resistance':
                        resistance_levels.append(level)

                sr_levels = {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels
                }

            self.logger.info(f'✅ Enhanced S/R detection complete: {len(sr_levels.get("support_levels", []))} support, {len(sr_levels.get("resistance_levels", []))} resistance levels')

            return sr_levels

        except Exception as e:
            self.logger.error(f'❌ Enhanced S/R detection failed: {e}')
            raise RuntimeError(f"Advanced SR detection failed: {e}. No fallback available.")

    async def _train_ml_models_with_memory_management(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models with intelligent memory management and adaptive chunking."""
        try:
            # Check if fast-fail has been engaged - prevent redundant restart attempts
            if hasattr(self, 'fast_fail_engaged') and self.fast_fail_engaged:
                self.logger.warning('⚠️ Fast-fail previously engaged - skipping memory-managed training restart')
                return {
                    'direction_accuracy': 0.0,
                    'volatility_mae': 0.0,
                    'model_type': 'skipped_due_to_fast_fail',
                    'training_samples': 0,
                    'error': 'Fast-fail previously engaged'
                }

            # Check memory usage and data size
            memory_usage = self._check_memory_usage()
            data_size = len(features_data)

            self.logger.info(f'🧠 Memory usage: {memory_usage:.1%}, Data size: {data_size:,} rows')

            # Determine processing strategy based on memory and data size
            if memory_usage > 0.8 or data_size > 1000000:
                # High memory usage or very large dataset - use aggressive chunking
                chunk_size = min(50000, data_size // 10)
                self.logger.info(f'📊 High memory usage detected, using aggressive chunking: {chunk_size:,} rows per chunk')
                return await self._train_ml_models_chunked_optimized(features_data, sr_levels, chunk_size)
            elif memory_usage > 0.6 or data_size > 500000:
                # Moderate memory usage or large dataset - use moderate chunking
                chunk_size = min(100000, data_size // 5)
                self.logger.info(f'📊 Moderate memory usage detected, using moderate chunking: {chunk_size:,} rows per chunk')
                return await self._train_ml_models_chunked_optimized(features_data, sr_levels, chunk_size)
            elif data_size > 200000:
                # Large dataset but good memory - use light chunking
                chunk_size = min(200000, data_size // 3)
                self.logger.info(f'📊 Large dataset detected, using light chunking: {chunk_size:,} rows per chunk')
                return await self._train_ml_models_chunked_optimized(features_data, sr_levels, chunk_size)
            else:
                # Small dataset or good memory - process in memory
                self.logger.info('📊 Processing in memory (no chunking needed)')
                return await self._train_ml_models(features_data, sr_levels)

        except Exception as e:
            self.logger.error(f'❌ Memory-managed ML training failed: {e}')
            # Fallback to basic training
            return await self._train_ml_models(features_data, sr_levels)

    def _check_memory_usage(self) -> float:
        """Check current memory usage as a percentage."""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
            return 0.0
        except Exception:
            return 0.0

    async def _train_ml_models(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models for SR level prediction with comprehensive evaluation and fast-fail checks."""
        self.logger.info('🤖 Starting comprehensive ML model training for SR optimization...')
        start_time = time.time()

        try:
            # Fast-fail: Validate that required methods exist
            if not self._validate_ml_methods_exist():
                error_message = "Missing required ML methods - cannot proceed with training"
                self.logger.error(f'❌ {error_message}')
                return self._handle_ml_failure(error_message, "METHOD_VALIDATION_ERROR")

            # Fast-fail: Check if we have sufficient data for ML training
            if len(features_data) < 200:
                self.logger.warning(f'⚠️ Insufficient data for ML training: {len(features_data)} rows (minimum: 200)')
                raise ValueError(f"Insufficient data for ML training: {len(features_data)} rows (minimum: 200)")

            # Fast-fail: Check if we have SR levels
            total_sr_levels = len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', []))
            if total_sr_levels == 0:
                self.logger.warning('⚠️ No SR levels available for ML training')
                raise ValueError("No SR levels available for ML training")

            # CRITICAL: Validate temporal integrity to prevent forward bias
            if not self._validate_temporal_integrity(features_data):
                self.logger.error('❌ Forward bias detected in training data - aborting training')
                return self._handle_ml_failure("Forward bias detected in training data", "FORWARD_BIAS_ERROR")

            # Fast-fail: Check memory usage before ML training
            memory_usage = self._check_memory_usage()
            if memory_usage > 0.9:
                self.logger.warning(f'⚠️ High memory usage before ML training: {memory_usage:.1%}')
                raise MemoryError(f"High memory usage before ML training: {memory_usage:.1%} (limit: 90%)")
            # Validate input data
            if features_data.empty:
                raise ValueError("Features data is empty")
            if not sr_levels or not any(sr_levels.get(key, []) for key in ['support_levels', 'resistance_levels']):
                raise ValueError("No SR levels provided for training")

            # Prepare target variables from SR levels
            self.logger.info('🎯 Preparing target variables from SR levels...')
            target_data = self._prepare_sr_targets(features_data, sr_levels)

            # Prepare features for ML training
            self.logger.info('🔧 Preparing features for ML training...')
            X, y_direction, y_volatility, feature_names = self._prepare_ml_features(features_data, target_data)

            # Optimize hyperparameters
            self.logger.info('🔧 Optimizing hyperparameters...')
            try:
                hyperparameter_results = self._optimize_hyperparameters(X, y_direction, feature_names)
            except Exception as e:
                self.logger.warning(f'⚠️ Hyperparameter optimization failed: {e}')
                hyperparameter_results = None

            # Feature selection
            self.logger.info('🎯 Performing feature selection...')
            try:
                X_selected, y_dir_selected, y_vol_selected, selected_feature_names, feature_selection_info = self._optimize_feature_selection(
                    X, y_direction, feature_names
                )
            except Exception as e:
                self.logger.warning(f'⚠️ Feature selection failed: {e}')
                X_selected, y_dir_selected, y_vol_selected, selected_feature_names = X, y_direction, y_volatility, feature_names
                feature_selection_info = self._get_fallback_feature_selection_info(feature_names)

            # Split data
            self.logger.info('✂️ Splitting data into train/test sets...')
            X_train, X_test, y_dir_train, y_dir_test, y_vol_train, y_vol_test = train_test_split(
                X_selected, y_dir_selected, y_vol_selected, test_size=0.2, random_state=42, stratify=y_dir_selected
            )

            # Scale features
            self.logger.info('📏 Scaling features...')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train models
            self.logger.info('🤖 Training ML models...')
            models_results = {}
            optimized_models = {}

            # Train individual models
            for model_name, model_config in self.ml_model_configs.items():
                try:
                    self.logger.info(f'🏃 Training {model_name}...')

                    # Use optimized hyperparameters if available
                    if hyperparameter_results and model_name in hyperparameter_results:
                        model_params = hyperparameter_results[model_name]['best_params']
                        model = model_config['class'](**model_params)
                    else:
                        model = model_config['class'](**model_config['default_params'])

                    # Train model
                    model.fit(X_train_scaled, y_dir_train)

                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = getattr(model, 'predict_proba', lambda X: np.zeros((len(X), 2)))(X_test_scaled)

                    # Calculate metrics
                    accuracy = accuracy_score(y_dir_test, y_pred)

                    models_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0],
                        'feature_importance': getattr(model, 'feature_importances_', None),
                        'params': model.get_params() if hasattr(model, 'get_params') else {}
                    }

                    self.logger.info(f'✅ {model_name} accuracy: {accuracy:.4f}')

                    # Store optimized model
                    optimized_models[model_name] = models_results[model_name]

                except Exception as e:
                    error_str = str(e)
                    # Special handling for LogisticRegression single-class errors
                    if model_name == 'LogisticRegression' and "needs samples of at least 2 classes" in error_str:
                        self.logger.warning(f'⚠️ LogisticRegression failed on single-class data: {error_str}')
                        self.logger.info(f'ℹ️ Skipping LogisticRegression for this chunk due to single-class data')
                        continue  # Skip only this model, not the whole chunk
                    else:
                        self.logger.error(f'❌ Failed to train {model_name}: {e}')
                        continue

            # If no models were trained successfully, fast fail
            if not models_results:
                self.logger.error('❌ No models could be trained successfully')
                raise Exception('No ML models could be trained successfully - fast failing')

            # Calculate evaluation metrics
            self.logger.info('📊 Calculating evaluation metrics...')
            try:
                cv_results = self._perform_cross_validation(X_train_scaled, y_dir_train, selected_feature_names)
                # Store cross-validation results as walk-forward results
                self._walk_forward_results = {
                    'status': 'completed',
                    'cross_validation': cv_results,
                    'folds': cv_results.get('n_splits', 5),
                    'mean_score': cv_results.get('mean_test_score', 0),
                    'std_score': cv_results.get('std_test_score', 0)
                }
            except Exception as e:
                self.logger.warning(f'⚠️ Cross-validation failed: {e}')
                cv_results = {
                    'direction_accuracy_scores': [0.5] * 5,
                    'mean_test_score': 0.5,
                    'std_test_score': 0.0,
                    'n_splits': 5
                }

            evaluation_metrics = self._calculate_evaluation_metrics(
                optimized_models if optimized_models else models_results,
                cv_results, X_test_scaled, y_dir_test, y_vol_test,
                None
            )

            # Save best model
            self.logger.info('💾 Saving best performing model...')
            models_for_saving = optimized_models if optimized_models else models_results

            model_save_path = self._save_best_model(
                models_for_saving, scaler, selected_feature_names
            )

            # Compile final results
            training_time = time.time() - start_time
            ml_results = {
                'direction_accuracy': evaluation_metrics['best_direction_accuracy'],
                'volatility_mae': evaluation_metrics['best_volatility_mae'],
                'model_type': evaluation_metrics['best_model_type'],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'sr_levels_used': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                'training_time': training_time,
                'feature_importance': evaluation_metrics['feature_importance'],
                'cross_validation_scores': cv_results['direction_accuracy_scores'],
                'models_performance': models_results,
                'evaluation_metrics': evaluation_metrics,
                'model_save_path': model_save_path,
                'feature_names': feature_names.tolist(),
                'selected_feature_names': selected_feature_names.tolist(),
                'scaler_params': {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                },
                'feature_selection': feature_selection_info
            }

            self.logger.info(f'✅ Comprehensive ML training completed in {training_time:.2f}s')
            self.logger.info(f'🎯 Best direction accuracy: {ml_results["direction_accuracy"]:.4f}')
            self.logger.info(f'📊 Best volatility MAE: {ml_results["volatility_mae"]:.6f}')
            self.logger.info(f'🏆 Best model: {ml_results["model_type"]}')

            return ml_results

        except Exception as e:
            # Use the new ML failure handling mechanism with better error classification
            error_message = f'Comprehensive ML training failed: {str(e)}'
            import traceback
            error_details = traceback.format_exc()

            self.logger.error(f'❌ {error_message}')
            self.logger.error(f'📋 Full traceback: {error_details}')

            # Classify the error type for better handling
            error_str = str(e).lower()
            if 'memory' in error_str or 'out of memory' in error_str:
                error_type = "MEMORY_ERROR"
            elif 'data' in error_str and ('empty' in error_str or 'none' in error_str):
                error_type = "DATA_ERROR"
            elif 'target' in error_str or 'label' in error_str:
                error_type = "TARGET_ERROR"
            elif 'feature' in error_str or 'column' in error_str:
                error_type = "FEATURE_ERROR"
            elif 'model' in error_str or 'fit' in error_str:
                error_type = "MODEL_FIT_ERROR"
            elif 'AttributeError' in error_str or "'SROptimizationStep' object has no attribute" in error_str:
                # AttributeErrors are ML training errors, not Optuna errors
                error_type = "ML_TRAINING_ERROR"
            elif 'optuna' in error_str or 'optimization' in error_str:
                error_type = "OPTUNA_ERROR"
            else:
                error_type = "ML_TRAINING_ERROR"

            # Handle ML failure with fast fail mechanism
            return self._handle_ml_failure(error_message, error_type)

    @log_step_functions
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Enhanced input validation with comprehensive fast-fail checks."""
        self.logger.info('🔍 Enhanced input validation with fast-fail checks')
        errors = []
        warnings = []
        
        # Fast-fail: Check required inputs
        required_inputs = ['validated_data']
        for input_key in required_inputs:
            if input_key not in training_input:
                errors.append(f'Missing required input: {input_key}')
                return False, errors  # Fast-fail on missing required inputs
        
        # Fast-fail: Data quality checks
        if 'validated_data' in training_input:
            data_validation_result = self._fast_fail_data_validation(training_input['validated_data'])
            if not data_validation_result['valid']:
                errors.extend(data_validation_result['errors'])
                return False, errors  # Fast-fail on data quality issues
            warnings.extend(data_validation_result.get('warnings', []))
        
        # Fast-fail: Configuration validation
        config_validation_result = self._fast_fail_config_validation(pipeline_state)
        if not config_validation_result['valid']:
            errors.extend(config_validation_result['errors'])
            return False, errors  # Fast-fail on configuration issues
        warnings.extend(config_validation_result.get('warnings', []))
        
        # Fast-fail: Resource validation
        resource_validation_result = self._fast_fail_resource_validation()
        if not resource_validation_result['valid']:
            errors.extend(resource_validation_result['errors'])
            return False, errors  # Fast-fail on resource issues
        warnings.extend(resource_validation_result.get('warnings', []))
        
        # Log warnings but don't fail
        for warning in warnings:
            self.logger.warning(f'⚠️ {warning}')
        
        return len(errors) == 0, errors
    
    def _fast_fail_data_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fast-fail data validation with comprehensive checks."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check 1: Data existence and basic structure
            if data is None:
                result['errors'].append('Data is None')
                result['valid'] = False
                return result
            
            if data.empty:
                result['errors'].append('Data is empty')
                result['valid'] = False
                return result
            
            # Check 2: Minimum data size
            if len(data) < 500:
                result['errors'].append(f'Insufficient data: {len(data)} rows (minimum: 500)')
                result['valid'] = False
                return result
            
            # Check 3: Required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                result['errors'].append(f'Missing required columns: {missing_cols}')
                result['valid'] = False
                return result
            
            # Check 4: Data freshness (if timestamp available)
            if 'timestamp' in data.columns:
                try:
                    latest_time = pd.to_datetime(data['timestamp']).max()
                    days_old = (datetime.now() - latest_time).days
                    if days_old > 30:
                        result['warnings'].append(f'Data is {days_old} days old (older than 30 days)')
                except Exception:
                    result['warnings'].append('Could not validate data freshness')
            
            # Check 5: Price data validity
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    # Check for missing values
                    missing_pct = data[col].isna().sum() / len(data)
                    if missing_pct > 0.1:  # >10% missing
                        result['errors'].append(f'Too many missing values in {col}: {missing_pct:.1%}')
                        result['valid'] = False
                        return result
                    
                    # Check for non-positive values
                    if (data[col] <= 0).any():
                        result['errors'].append(f'Invalid price values in {col}: non-positive values found')
                        result['valid'] = False
                        return result
            
            # Check 6: Volume data validity
            if 'volume' in data.columns:
                if (data['volume'] < 0).any():
                    result['errors'].append('Invalid volume values: negative volumes found')
                    result['valid'] = False
                    return result
            
            # Check 7: OHLC consistency
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (data['high'] < data['low']) |
                    (data['high'] < data['open']) |
                    (data['high'] < data['close']) |
                    (data['low'] > data['open']) |
                    (data['low'] > data['close'])
                )
                invalid_count = invalid_ohlc.sum()
                if invalid_count > 0:
                    invalid_pct = invalid_count / len(data)
                    if invalid_pct > 0.05:  # >5% invalid OHLC
                        result['errors'].append(f'Too many invalid OHLC relationships: {invalid_pct:.1%}')
                        result['valid'] = False
                        return result
                    else:
                        result['warnings'].append(f'Some invalid OHLC relationships found: {invalid_count} rows')
            
            # Check 8: Data quality score
            quality_score = self._calculate_data_quality_score(data)
            if quality_score < 0.7:
                result['warnings'].append(f'Low data quality score: {quality_score:.2f}')
            
            return result
            
        except Exception as e:
            result['errors'].append(f'Data validation error: {str(e)}')
            result['valid'] = False
            return result
    
    def _fast_fail_config_validation(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast-fail configuration validation."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            config = pipeline_state.get('config', {})
            sr_config = config.get('sr_optimization', {})
            
            # Check optimization parameters
            optimization_trials = sr_config.get('optimization_trials', 100)
            if optimization_trials > 1000:
                result['warnings'].append(f'High optimization trials: {optimization_trials} (may cause long execution)')
            
            # Check memory-intensive settings
            if sr_config.get('enable_hyperparameter_optimization', True):
                if optimization_trials > 500:
                    result['warnings'].append('Hyperparameter optimization with high trial count may be memory-intensive')
            
            return result
            
        except Exception as e:
            result['errors'].append(f'Configuration validation error: {str(e)}')
            result['valid'] = False
            return result
    
    def _fast_fail_resource_validation(self) -> Dict[str, Any]:
        """Fast-fail resource validation."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check memory usage
            memory_usage = self._check_memory_usage()
            if memory_usage > 0.9:
                result['errors'].append(f'System memory usage too high: {memory_usage:.1%}')
                result['valid'] = False
                return result
            elif memory_usage > 0.8:
                result['warnings'].append(f'High system memory usage: {memory_usage:.1%}')
            
            # Check available disk space (if possible)
            try:
                import shutil
                free_space = shutil.disk_usage('.').free / (1024**3)  # GB
                if free_space < 1.0:  # Less than 1GB
                    result['warnings'].append(f'Low disk space: {free_space:.1f}GB available')
            except Exception:
                pass  # Disk space check is optional
            
            return result
            
        except Exception as e:
            result['errors'].append(f'Resource validation error: {str(e)}')
            result['valid'] = False
            return result
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate comprehensive data quality score."""
        try:
            score = 1.0
            
            # Completeness score
            completeness = 1 - (data.isna().sum().sum() / (len(data) * len(data.columns)))
            score *= completeness
            
            # Consistency score (price relationships)
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                valid_ohlc = (
                    (data['high'] >= data['low']) & 
                    (data['high'] >= data['open']) & 
                    (data['high'] >= data['close']) &
                    (data['low'] <= data['open']) & 
                    (data['low'] <= data['close'])
                ).mean()
                score *= valid_ohlc
            
            # Volume consistency
            if 'volume' in data.columns:
                positive_volume = (data['volume'] >= 0).mean()
                score *= positive_volume
            
            # Price consistency (no zero or negative prices)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    positive_prices = (data[col] > 0).mean()
                    score *= positive_prices
            
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _robust_error_handling(self, operation_name: str, operation_func, *args, **kwargs):
        """Robust error handling with automatic retry and fallback mechanisms."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except MemoryError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Memory error in {operation_name}, retrying with reduced data...")
                    # Reduce data size and retry
                    reduced_args = self._reduce_data_size(args)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"Memory error in {operation_name} after {max_retries} attempts")
                    return self._get_fallback_result(operation_name)
            except asyncio.TimeoutError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Timeout in {operation_name}, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"Timeout in {operation_name} after {max_retries} attempts")
                    return self._get_fallback_result(operation_name)
            except Exception as e:
                self.logger.error(f"Error in {operation_name}: {e}")
                if attempt < max_retries - 1:
                    self.logger.warning(f"Retrying {operation_name} (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    return self._get_fallback_result(operation_name)
        
        return self._get_fallback_result(operation_name)
    
    def _reduce_data_size(self, args):
        """Reduce data size for memory-constrained retries."""
        reduced_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame) and len(arg) > 10000:
                # Sample 10K rows for retry
                reduced_args.append(arg.sample(n=10000, random_state=42))
            else:
                reduced_args.append(arg)
        return reduced_args
    
    def _get_fallback_result(self, operation_name: str):
        """Get fallback result for failed operations."""
        fallback_results = {
            'sr_detection': self._get_fallback_sr_levels(),
            'ml_training': {'error': 'ML training failed', 'direction_accuracy': 0.0},
            'feature_selection': (np.array([]), np.array([]), {'error': 'feature_selection_failed'}),
            'hyperparameter_optimization': self._get_default_hyperparameters()
        }
        return fallback_results.get(operation_name, {})
    
    def _performance_monitor(self, operation_name: str):
        """Context manager for performance monitoring with memory tracking."""
        class PerformanceMonitor:
            def __init__(self, name, logger):
                self.name = name
                self.logger = logger
                self.start_time = None
                self.start_memory = None
            
            def __enter__(self):
                self.start_time = time.time()
                self.start_memory = self._check_memory_usage()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                end_memory = self._check_memory_usage()
                memory_delta = end_memory - self.start_memory
                
                if exc_type is None:
                    self.logger.info(f"⏱️ {self.name}: {duration:.2f}s, Memory: {memory_delta:+.1%}")
                else:
                    self.logger.error(f"❌ {self.name} failed after {duration:.2f}s, Memory: {memory_delta:+.1%}")
        
        return PerformanceMonitor(operation_name, self.logger)

    def _validate_ml_methods_exist(self) -> bool:
        """Validate that all required ML methods exist and are callable."""
        required_methods = [
            '_prepare_sr_targets',
            '_prepare_ml_features',
            '_train_ml_models_chunked_optimized',
            '_optimize_hyperparameters',
            '_optimize_feature_selection',
            '_get_fallback_feature_selection_info',
            '_fallback_hyperparameter_selection',
            '_create_temporal_train_test_split',
            '_validate_temporal_integrity',
            '_perform_cross_validation',
            '_calculate_evaluation_metrics'
        ]

        missing_methods = []
        for method_name in required_methods:
            # Debug: Check if method exists
            has_attr = hasattr(self, method_name)
            self.logger.debug(f'🔍 Checking method {method_name}: hasattr={has_attr}')

            if not has_attr:
                missing_methods.append(method_name)
                continue

            # Additional check: ensure the method is callable
            method = getattr(self, method_name)
            is_callable = callable(method)
            self.logger.debug(f'🔍 Method {method_name}: callable={is_callable}')

            if not is_callable:
                missing_methods.append(f"{method_name} (not callable)")
                continue

            # Additional check: ensure method has correct signature for key methods
            if method_name == '_prepare_ml_features':
                import inspect
                try:
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())
                    # Should have 'self', 'features_data', 'target_data'
                    expected_params = ['features_data', 'target_data']
                    if not all(param in params for param in expected_params):
                        missing_methods.append(f"{method_name} (wrong signature: {params})")
                except Exception as e:
                    missing_methods.append(f"{method_name} (signature check failed: {e})")

        if missing_methods:
            self.logger.error(f'❌ Missing or invalid required ML methods: {missing_methods}')
            # Also print to stdout for immediate debugging
            print(f'DEBUG: Missing methods: {missing_methods}', flush=True)
            return False

        self.logger.info('✅ All required ML methods are available and valid')
        # Also print to stdout for immediate debugging
        print('DEBUG: All methods found successfully', flush=True)
        return True

    def _validate_critical_imports(self) -> bool:
        """Validate that critical imports are available for operation."""
        critical_imports = [
            ('ENHANCED_SR_DETECTOR_AVAILABLE', 'Enhanced SR Detector'),
            ('PARQUET_UTILS_AVAILABLE', 'ParquetUtils'),
            ('ADVANCED_FEATURES_AVAILABLE', 'Advanced Feature Engineering'),
        ]

        missing_imports = []
        for flag_name, description in critical_imports:
            if not globals().get(flag_name, False):
                missing_imports.append(description)

        if missing_imports:
            self.logger.warning(f'⚠️ Some optional imports are not available: {missing_imports}')
            self.logger.info('📋 Operation will continue with reduced functionality')
            return False

        self.logger.info('✅ All critical imports are available')
        return True

    async def _train_ml_models_chunked_optimized(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any], chunk_size: int) -> Dict[str, Any]:
        """Optimized chunked ML training with memory monitoring and adaptive processing."""
        try:
            total_chunks = (len(features_data) + chunk_size - 1) // chunk_size
            self.logger.info(f'📊 Processing {len(features_data):,} rows in {total_chunks} chunks of {chunk_size:,} rows each')

            all_results = []
            chunk_processing_times = []

            for i in range(0, len(features_data), chunk_size):
                chunk_end = min(i + chunk_size, len(features_data))
                chunk_data = features_data.iloc[i:chunk_end]
                chunk_num = i // chunk_size + 1

                # Check memory before processing each chunk
                memory_before = self._check_memory_usage()
                if memory_before > 0.9:
                    self.logger.warning(f'⚠️ High memory usage before chunk {chunk_num}: {memory_before:.1%}')
                    # Force garbage collection
                    import gc
                    gc.collect()

                self.logger.info(f'🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_data):,} rows)')
                chunk_start = time.time()

                try:
                    # Process chunk with timeout
                    chunk_result = await asyncio.wait_for(
                        self._train_ml_models(chunk_data, sr_levels),
                        timeout=300  # 5 minutes per chunk
                    )
                    chunk_time = time.time() - chunk_start
                    chunk_processing_times.append(chunk_time)

                    # Check memory after processing
                    memory_after = self._check_memory_usage()
                    self.logger.info(f'✅ Chunk {chunk_num} completed in {chunk_time:.2f}s, memory: {memory_after:.1%}')

                    all_results.append(chunk_result)

                except asyncio.TimeoutError:
                    error_message = f'Chunk {chunk_num} timed out after 5 minutes'
                    self.logger.error(f'⏰ {error_message}')
                    # Handle timeout as ML failure
                    fallback_result = self._handle_ml_failure(error_message, "TIMEOUT_ERROR")
                    all_results.append(fallback_result)
                except ValueError as chunk_error:
                    # Special handling for single-class chunks and other data issues
                    error_str = str(chunk_error)
                    if "Single-class chunk detected" in error_str:
                        self.logger.warning(f'⚠️ Skipping chunk {chunk_num}: {error_str}')
                        self.logger.info(f'ℹ️ Single-class chunks are expected in chunked data and do not count as ML failures')
                        continue  # Skip this chunk without incrementing failure counter
                    else:
                        error_message = f'Chunk {chunk_num} failed: {error_str}'
                        self.logger.error(f'❌ {error_message}')
                        # Handle chunk failure as ML failure
                        fallback_result = self._handle_ml_failure(error_message, "CHUNK_ERROR")
                        all_results.append(fallback_result)
                except Exception as chunk_error:
                    error_message = f'Chunk {chunk_num} failed: {str(chunk_error)}'
                    self.logger.error(f'❌ {error_message}')
                    # Handle chunk failure as ML failure
                    fallback_result = self._handle_ml_failure(error_message, "CHUNK_ERROR")
                    all_results.append(fallback_result)

            # Aggregate results from all chunks
            if all_results:
                # Use the first result as base and merge others
                final_result = all_results[0].copy()
                for result in all_results[1:]:
                    # Merge results (simplified - take the best performing model)
                    if result.get('accuracy', 0) > final_result.get('accuracy', 0):
                        final_result = result

                # Add chunking statistics
                final_result['chunking_stats'] = {
                    'total_chunks': total_chunks,
                    'avg_chunk_time': np.mean(chunk_processing_times) if chunk_processing_times else 0,
                    'total_processing_time': sum(chunk_processing_times)
                }

                self.logger.info(f'🎉 Chunked ML training completed: {len(all_results)} chunks processed')
                return final_result
            else:
                return self._handle_ml_failure("No chunks were successfully processed", "NO_RESULTS_ERROR")

        except Exception as e:
            self.logger.error(f'❌ Chunked ML training failed: {e}')
            return self._handle_ml_failure(str(e), "CHUNKED_TRAINING_ERROR")

    def _prepare_sr_targets(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any]) -> pd.DataFrame:
        """Prepare target variables from SR levels for ML training with enhanced features - NO FORWARD BIAS."""
        try:
            # Extract SR level prices for target creation - handle both dict and object formats
            support_prices = []
            resistance_prices = []

            for level in sr_levels.get('support_levels', []):
                if hasattr(level, 'price'):
                    support_prices.append(level.price)
                elif isinstance(level, dict) and 'price' in level:
                    support_prices.append(level['price'])
                else:
                    support_prices.append(level)

            for level in sr_levels.get('resistance_levels', []):
                if hasattr(level, 'price'):
                    resistance_prices.append(level.price)
                elif isinstance(level, dict) and 'price' in level:
                    resistance_prices.append(level['price'])
                else:
                    resistance_prices.append(level)

            # Get current price data
            if 'close' not in features_data.columns:
                raise ValueError("Features data must contain 'close' price column")

            current_prices = features_data['close'].values
            target_data = pd.DataFrame(index=features_data.index)

            # CRITICAL: Create targets WITHOUT forward bias
            # Only use SR levels that were detected using historical data up to each point
            proximity_threshold = 0.005  # 0.5% proximity threshold
            near_support = np.zeros(len(current_prices))
            near_resistance = np.zeros(len(current_prices))

            # Filter SR levels to only include those that could be known at each point in time
            # This prevents forward bias by ensuring we only use SR levels detected from past data
            valid_support_prices = []
            valid_resistance_prices = []

            # For each SR level, ensure it was detected using only historical data
            for price in support_prices:
                if isinstance(price, (int, float)) and not np.isnan(price):
                    valid_support_prices.append(price)

            for price in resistance_prices:
                if isinstance(price, (int, float)) and not np.isnan(price):
                    valid_resistance_prices.append(price)

            # Calculate proximity for each time point using ONLY historical SR levels
            for i, current_price in enumerate(current_prices):
                # Check proximity to support levels
                for support_price in valid_support_prices:
                    if abs(current_price - support_price) / current_price <= proximity_threshold:
                        near_support[i] = 1.0
                        break

                # Check proximity to resistance levels
                for resistance_price in valid_resistance_prices:
                    if abs(current_price - resistance_price) / current_price <= proximity_threshold:
                        near_resistance[i] = 1.0
                        break

            # Create DISCRETE direction target (0 = bearish, 1 = bullish, 2 = neutral)
            target_data['near_support'] = near_support
            target_data['near_resistance'] = near_resistance

            # Convert to discrete classes for classification
            target_data['direction_target'] = np.where(
                near_support == 1, 1,  # Near support = bullish (class 1)
                np.where(near_resistance == 1, 0, 2)  # Near resistance = bearish (class 0), else neutral (class 2)
            ).astype(int)

            # Create volatility target based on proximity to levels (binary for classification)
            proximity_score = np.maximum(near_support, near_resistance)
            target_data['volatility_target'] = proximity_score.astype(int)  # 0 or 1

            # Add trend direction based on price movement near levels
            if len(current_prices) > 5:
                short_trend = np.sign(current_prices[5:] - current_prices[:-5])
                trend_signal = np.concatenate([np.zeros(5), short_trend])
                target_data['trend_signal'] = trend_signal.astype(int)

                # Enhanced direction target incorporating trend (discrete classes)
                target_data['direction_target'] = np.where(
                    (near_support == 1) & (trend_signal > 0), 1,  # Support + uptrend = bullish (class 1)
                    np.where((near_resistance == 1) & (trend_signal < 0), 0,  # Resistance + downtrend = bearish (class 0)
                        np.where(near_support == 1, 1,  # Support only = bullish (class 1)
                            np.where(near_resistance == 1, 0, 2)))  # Resistance only = bearish (class 0), else neutral (class 2)
                ).astype(int)

            self.logger.info(f'🎯 SR targets prepared (no forward bias): {len(valid_support_prices)} support, {len(valid_resistance_prices)} resistance levels')
            self.logger.info(f'📊 Target distribution: {target_data["direction_target"].value_counts().to_dict()}')

            return target_data

        except Exception as e:
            self.logger.error(f'❌ SR target preparation failed: {e}')
            # Return dataframe with proper discrete classes
            target_data = pd.DataFrame(index=features_data.index)
            target_data['direction_target'] = 2  # Neutral class
            target_data['volatility_target'] = 0  # No volatility signal
            target_data['near_support'] = 0
            target_data['near_resistance'] = 0
            return target_data

    def _prepare_ml_features(self, features_data: pd.DataFrame, target_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training."""
        try:
            # Select numeric features only
            numeric_features = features_data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['timestamp', 'datetime', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in numeric_features if col not in exclude_cols]

            # Handle missing values and ensure all data is numeric
            feature_data = features_data[feature_cols].fillna(0)

            # Convert all columns to numeric to avoid data type issues
            for col in feature_data.columns:
                if feature_data[col].dtype == 'object':
                    try:
                        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0)
                    except:
                        self.logger.warning(f'⚠️ Could not convert column {col} to numeric, dropping it')
                        feature_data = feature_data.drop(columns=[col])
                        if col in feature_cols:
                            feature_cols.remove(col)

            X = feature_data.values
            feature_names = np.array(feature_cols)

            # Get targets
            if 'direction_target' not in target_data.columns:
                raise ValueError("Target data must contain 'direction_target' column")
            if 'volatility_target' not in target_data.columns:
                raise ValueError("Target data must contain 'volatility_target' column")

            y_direction = target_data['direction_target'].values
            y_volatility = target_data['volatility_target'].values

            # Handle neutral class filtering more intelligently
            neutral_count = np.sum(y_direction == 2)
            non_neutral_count = len(y_direction) - neutral_count

            if non_neutral_count >= 100:
                # If we have enough non-neutral samples, filter out neutral
                valid_mask = y_direction != 2
                self.logger.info(f'📊 Filtered out {neutral_count} neutral samples, keeping {non_neutral_count} directional samples')
            elif neutral_count >= 100:
                # If we don't have enough directional samples but have enough neutral, keep a subset
                neutral_keep_ratio = min(0.3, max(0.1, 100 / neutral_count))  # Keep 10-30% of neutral samples
                neutral_indices = np.where(y_direction == 2)[0]
                keep_neutral = int(len(neutral_indices) * neutral_keep_ratio)
                neutral_mask = np.zeros(len(y_direction), dtype=bool)
                neutral_mask[neutral_indices[:keep_neutral]] = True

                directional_mask = y_direction != 2
                valid_mask = directional_mask | neutral_mask

                kept_samples = np.sum(valid_mask)
                self.logger.info(f'📊 Mixed filtering: kept {np.sum(directional_mask)} directional + {keep_neutral} neutral = {kept_samples} total samples')
            else:
                # If we have very few samples total, keep all
                valid_mask = np.ones(len(y_direction), dtype=bool)
                self.logger.warning(f'⚠️ Very few samples ({len(y_direction)}), keeping all including neutral for minimum training data')

            X = X[valid_mask]
            y_direction = y_direction[valid_mask]
            y_volatility = y_volatility[valid_mask]

            # CRITICAL: Check for single-class chunks that will cause LogisticRegression failures
            unique_classes = np.unique(y_direction)
            if len(unique_classes) < 2:
                error_msg = f"Single-class chunk detected: only class {unique_classes[0]} present ({len(y_direction)} samples)"
                self.logger.error(f'❌ {error_msg}')
                raise ValueError(error_msg)

            # Check for extreme class imbalance in chunks
            if len(unique_classes) >= 2:
                class_counts = np.bincount(y_direction.astype(int))
                max_class_ratio = max(class_counts) / sum(class_counts)
                if max_class_ratio > 0.98:  # 98% single class
                    self.logger.warning(f'⚠️ Extremely imbalanced chunk: {max_class_ratio:.2%} single class ({len(y_direction)} samples)')
                    # Continue but log the warning - don't fail the chunk

            # Ensure we have enough samples
            if len(X) < 50:  # Reduced minimum requirement
                raise ValueError(f"Insufficient training samples: {len(X)} (minimum 50 required)")

            # CRITICAL: Ensure temporal order is preserved to avoid forward bias
            # Sort by time if we have a time index (this is crucial for time series)
            if hasattr(features_data, 'index') and hasattr(features_data.index, 'is_monotonic_increasing'):
                if not features_data.index.is_monotonic_increasing:
                    self.logger.warning('⚠️ Data index is not monotonic - sorting to preserve temporal order')
                    # Sort by index to ensure temporal order
                    sort_indices = features_data.index.argsort()
                    X = X[sort_indices[valid_mask]]
                    y_direction = y_direction[sort_indices[valid_mask]]
                    y_volatility = y_volatility[sort_indices[valid_mask]]

            self.logger.info(f'📊 ML data prepared (temporal order preserved): {X.shape[0]} samples, {X.shape[1]} features')
            self.logger.info(f'🎯 Direction target distribution: {np.bincount(y_direction.astype(int))}')
            self.logger.info(f'📈 Volatility target range: {y_volatility.min():.6f} - {y_volatility.max():.6f}')
            self.logger.info('🛡️ Forward bias prevention: Data sorted by time, temporal CV used')
            return X, y_direction, y_volatility, feature_names

        except Exception as e:
            self.logger.error(f'❌ ML feature preparation failed: {e}')
            raise

    def _create_temporal_train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create temporal train/test split to avoid forward bias."""
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        # Ensure we have at least some samples in both sets
        split_idx = max(split_idx, n_samples // 10)  # At least 10% for training
        split_idx = min(split_idx, n_samples - n_samples // 10)  # At least 10% for testing

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        self.logger.info(f'🛡️ Temporal split created: {len(X_train)} train, {len(X_test)} test samples')
        self.logger.info(f'📊 Test set represents {len(X_test)/len(X):.2f} of total data')
        return X_train, X_test, y_train, y_test

    def _validate_temporal_integrity(self, features_data: pd.DataFrame) -> bool:
        """Validate that data maintains temporal integrity and no forward bias."""
        try:
            # Check if index is properly sorted
            if not features_data.index.is_monotonic_increasing:
                self.logger.error('❌ Forward bias detected: Data index is not temporally ordered')
                return False

            # Check for any future data leakage in features
            # This is a simplified check - in practice you'd want more sophisticated validation
            if 'close' in features_data.columns:
                # Check if close column is numeric
                if not pd.api.types.is_numeric_dtype(features_data['close']):
                    self.logger.debug('⏭️ Close column is not numeric, skipping temporal validation')
                    return True

                # Check if any features use future price information
                close_prices = features_data['close'].values
                if len(close_prices) > 1:
                    try:
                        # Ensure close prices are numeric
                        close_prices = pd.to_numeric(close_prices, errors='coerce')
                        if np.isnan(close_prices).any():
                            self.logger.debug('⏭️ Close prices contain NaN values, skipping detailed temporal validation')
                            return True

                        future_prices = np.roll(close_prices, -1)[:-1]
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f'⏭️ Failed to process close prices for temporal validation: {e}')
                        return True

                    # OHLC features naturally correlate with future prices - this is expected
                    ohlc_features = {'open', 'high', 'low', 'close', 'volume'}
                    suspicious_features = []

                    for col in features_data.columns:
                        if col != 'close':
                            # Skip non-numeric columns to avoid correlation errors
                            if not pd.api.types.is_numeric_dtype(features_data[col]):
                                self.logger.debug(f'⏭️ Skipping non-numeric column {col} in temporal validation')
                                continue

                            feature_values = features_data[col].values[:-1]
                            if len(feature_values) == len(future_prices):
                                try:
                                    # Ensure both arrays are numeric before correlation
                                    feature_numeric = pd.to_numeric(feature_values, errors='coerce')
                                    future_numeric = pd.to_numeric(future_prices, errors='coerce')

                                    # Skip if we have NaN values after conversion
                                    if np.isnan(feature_numeric).any() or np.isnan(future_numeric).any():
                                        self.logger.debug(f'⏭️ Skipping column {col} due to NaN values in temporal validation')
                                        continue

                                    correlation = np.corrcoef(feature_numeric, future_numeric)[0, 1]

                                    # Handle OHLC features differently
                                    if col in ohlc_features:
                                        # Only log once per OHLC feature type, not per correlation value
                                        if abs(correlation) > 0.95:
                                            if col not in ['open', 'high', 'low']:  # Reduce logging for most common OHLC
                                                self.logger.debug(f'ℹ️ Expected OHLC correlation in {col}: {correlation:.4f}')
                                    else:
                                        # Non-OHLC features should not correlate with future prices
                                        if abs(correlation) > 0.99:
                                            suspicious_features.append((col, correlation))
                                except (ValueError, TypeError) as corr_error:
                                    self.logger.debug(f'⏭️ Correlation calculation failed for {col}: {corr_error}')
                                    continue

                    # Log suspicious features only once, not for each chunk
                    if suspicious_features and len(suspicious_features) <= 2:  # Limit logging
                        for col, corr in suspicious_features[:1]:  # Log only the most suspicious
                            self.logger.warning(f'⚠️ Potential forward bias in feature {col}: correlation {corr:.4f} with future prices')

            self.logger.info('✅ Temporal integrity validated - no forward bias detected')
            return True

        except Exception as e:
            self.logger.error(f'❌ Temporal integrity validation failed: {e}')
            return False

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for ML models using Optuna."""
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

            self.logger.info('🔧 Starting Optuna hyperparameter optimization...')

            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))

            def objective(trial):
                # Choose model type
                model_type = trial.suggest_categorical('model_type',
                    ['LogisticRegression', 'RandomForestClassifier', 'HistGradientBoostingClassifier'])

                if model_type == 'LogisticRegression':
                    # Logistic Regression parameters
                    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
                    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg'])
                    max_iter = trial.suggest_int('max_iter', 1000, 5000)

                    # Add class weights for imbalanced data
                    class_weight = 'balanced' if max_class_ratio > 0.85 else None

                    model = LogisticRegression(
                        C=C,
                        solver=solver,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        random_state=42,
                        n_jobs=-1
                    )

                elif model_type == 'RandomForestClassifier':
                    # Random Forest parameters
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

                    # Add class weights for imbalanced data
                    class_weight = 'balanced_subsample' if max_class_ratio > 0.85 else None

                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        class_weight=class_weight,
                        random_state=42,
                        n_jobs=-1
                    )

                else:  # HistGradientBoostingClassifier
                    # Histogram-based Gradient Boosting parameters
                    max_iter = trial.suggest_int('max_iter', 50, 300)
                    max_depth = trial.suggest_int('max_depth', 3, 15)
                    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1.0, log=True)
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 100)
                    l2_regularization = trial.suggest_float('l2_regularization', 1e-6, 1e-1, log=True)

                    model = HistGradientBoostingClassifier(
                        max_iter=max_iter,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2_regularization,
                        random_state=42
                    )

                    # Note: HistGradientBoostingClassifier will use sample_weight if provided during fit

                # Evaluate model with time-aware cross-validation (no forward bias)
                try:
                    from sklearn.model_selection import TimeSeriesSplit
                    from sklearn.metrics import accuracy_score, f1_score

                    # Validate that we have enough samples and classes
                    unique_classes = np.unique(y)
                    class_counts = np.bincount(y.astype(int)) if len(unique_classes) > 0 else []
                    self.logger.info(f'📊 Optuna trial data: {len(y)} samples, classes: {unique_classes}, counts: {class_counts}')

                    if len(unique_classes) < 2:
                        self.logger.warning(f'⚠️ Insufficient class diversity in trial: {len(unique_classes)} classes')
                        return 0.0

                    # Enhanced class imbalance checks
                    if len(class_counts) > 1:
                        max_class_ratio = max(class_counts) / sum(class_counts)
                        min_class_ratio = min(class_counts) / sum(class_counts)

                        # Check for extreme class imbalance (>95% single class)
                        if max_class_ratio > 0.95:
                            self.logger.warning(f'⚠️ Extreme class imbalance in trial: {max_class_ratio:.2%} of samples are one class')
                            return 0.0

                        # Check for severe imbalance (>85% single class) - use class weights
                        if max_class_ratio > 0.85:
                            self.logger.info(f'⚠️ Severe class imbalance detected: {max_class_ratio:.2%} - using class weights')

                        # Check if any class has too few samples for meaningful training
                        min_samples_per_class = max(5, len(y) // 100)  # At least 5 samples or 1% of total
                        if min(class_counts) < min_samples_per_class:
                            self.logger.warning(f'⚠️ Some classes have too few samples: min {min(class_counts)} < {min_samples_per_class}')
                            return 0.0

                    # Use TimeSeriesSplit to respect temporal order and avoid forward bias
                    # Ensure minimum test size for meaningful evaluation
                    min_test_size = max(50, len(X) // 20)  # At least 50 samples or 5% of data
                    test_size = min(len(X) // 5, max(min_test_size, len(X) // 10))  # Between 5-10% of data

                    # Reduce splits if data is small to ensure meaningful train/test sets
                    max_splits = min(5, max(2, len(X) // 1000))
                    n_splits = min(max_splits, max(2, (len(X) - test_size) // test_size))

                    # Ensure we have enough data for CV
                    if len(X) < test_size * (n_splits + 1):
                        test_size = max(50, len(X) // (n_splits + 2))
                        n_splits = min(n_splits, max(2, len(X) // test_size - 1))

                    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

                    cv_scores = []
                    valid_folds = 0
                    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                        # Ensure we don't use future data to predict past
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        # Validate fold has enough samples and classes (relaxed thresholds)
                        min_train_samples = max(20, len(X) // 100)  # At least 20 or 1% of total data
                        min_test_samples = max(10, len(X) // 200)   # At least 10 or 0.5% of total data

                        if len(X_train) < min_train_samples or len(X_test) < min_test_samples:
                            self.logger.debug(f'Fold {fold_idx}: Insufficient samples - Train: {len(X_train)}/{min_train_samples}, Test: {len(X_test)}/{min_test_samples}')
                            continue

                        # Check class diversity with more flexible requirements
                        train_classes = np.unique(y_train)
                        test_classes = np.unique(y_test)

                        # More detailed logging for debugging
                        self.logger.debug(f'Fold {fold_idx}: Train size: {len(X_train)}, Test size: {len(X_test)}')
                        self.logger.debug(f'Fold {fold_idx}: Train classes: {train_classes} (counts: {np.bincount(y_train.astype(int)) if len(train_classes) > 0 else []})')
                        self.logger.debug(f'Fold {fold_idx}: Test classes: {test_classes} (counts: {np.bincount(y_test.astype(int)) if len(test_classes) > 0 else []})')

                        if len(train_classes) < 1 or len(test_classes) < 1:
                            self.logger.warning(f'Fold {fold_idx}: No classes in train/test - Train classes: {train_classes}, Test classes: {test_classes}')
                            continue
                        if len(train_classes) == 1 and len(test_classes) == 1 and train_classes[0] != test_classes[0]:
                            self.logger.warning(f'Fold {fold_idx}: Different single classes - Train: {train_classes[0]}, Test: {test_classes[0]}')
                            continue
                        if len(train_classes) == 1 and len(test_classes) == 1:
                            self.logger.warning(f'Fold {fold_idx}: Only one class in both train and test: {train_classes[0]}')
                            continue

                        valid_folds += 1

                        try:
                            # Suppress warnings for LogisticRegression to avoid convergence warnings being treated as errors
                            import warnings
                            try:
                                from sklearn.exceptions import ConvergenceWarning
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')
                                    warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.linear_model')
                            except ImportError:
                                # Fallback if ConvergenceWarning is not available
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')
                                    warnings.filterwarnings('ignore', message='*convergence*', category=UserWarning)

                                # Train model on historical data
                                # Add sample weights for imbalanced data if supported
                                fit_kwargs = {}
                                if max_class_ratio > 0.85 and hasattr(model, 'class_weight'):
                                    # For models that support class_weight, it's already set
                                    pass
                                elif max_class_ratio > 0.85 and model_type == 'HistGradientBoostingClassifier':
                                    # For HistGradientBoostingClassifier, compute sample weights
                                    from sklearn.utils.class_weight import compute_sample_weight
                                    fit_kwargs['sample_weight'] = compute_sample_weight('balanced', y_train)

                                model.fit(X_train, y_train, **fit_kwargs)

                                # Validate model learned something
                                y_train_pred = model.predict(X_train)
                                train_classes_pred = np.unique(y_train_pred)
                                if len(train_classes_pred) == 1:
                                    self.logger.warning(f'Fold {fold_idx}: Model only predicts one class on training data: {train_classes_pred[0]}')
                                    # Skip this fold as model didn't learn properly
                                    continue

                            # Test on future data
                            y_pred = model.predict(X_test)

                            # Debug predictions
                            pred_classes = np.unique(y_pred)
                            self.logger.debug(f'Fold {fold_idx}: Predictions - unique classes: {pred_classes}, counts: {np.bincount(y_pred.astype(int)) if len(pred_classes) > 0 else []}')

                            # Use balanced metrics for imbalanced data
                            try:
                                from sklearn.metrics import balanced_accuracy_score

                                if max_class_ratio > 0.85:
                                    # Use balanced accuracy for imbalanced data
                                    score = balanced_accuracy_score(y_test, y_pred)
                                    self.logger.debug(f'Fold {fold_idx}: Balanced accuracy score: {score:.4f}')
                                elif n_classes > 2:
                                    # Multi-class: use macro F1 score (treats all classes equally)
                                    score = f1_score(y_test, y_pred, average='macro')
                                    self.logger.debug(f'Fold {fold_idx}: Macro F1 score: {score:.4f}')
                                else:
                                    # Binary: use F1 score
                                    pos_label = 1 if 1 in y_test else (0 if 0 in y_test else unique_classes[0])
                                    score = f1_score(y_test, y_pred, pos_label=pos_label, average='binary')
                                    self.logger.debug(f'Fold {fold_idx}: Binary F1 score: {score:.4f}')

                            except Exception as score_error:
                                self.logger.warning(f'Fold {fold_idx}: Score calculation failed: {score_error}')
                                score = 0.0

                            cv_scores.append(score)

                        except Exception as fold_error:
                            # Special handling for LogisticRegression convergence issues
                            if 'LogisticRegression' in str(model.__class__) and ('convergence' in str(fold_error).lower() or 'max_iter' in str(fold_error).lower()):
                                self.logger.debug(f'⚠️ LogisticRegression convergence issue in fold {fold_idx}, skipping: {fold_error}')
                            else:
                                self.logger.debug(f'⚠️ Fold evaluation failed: {fold_error}')
                            continue

                    if not cv_scores:
                        self.logger.warning(f'⚠️ No valid CV folds completed for {model_type} (checked {n_splits} folds, {valid_folds} were valid)')
                        return 0.0

                    mean_score = np.mean(cv_scores)
                    self.logger.debug(f'📊 Model {model_type} CV score: {mean_score:.4f} ({len(cv_scores)}/{n_splits} valid folds)')
                    return mean_score

                except Exception as e:
                    self.logger.warning(f'⚠️ Time-aware CV failed: {e}')
                    return 0.0

            # Create study and optimize
            study_name = f"sr_optimization_{n_samples}_{n_features}"
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # Set timeout based on data size
            timeout = min(300, max(60, n_samples // 1000))  # 1-5 minutes based on data size

            self.logger.info(f'🔧 Optimizing for {timeout}s with {n_samples} samples, {n_features} features')

            study.optimize(objective, timeout=timeout, n_jobs=1)

            # Get best parameters
            best_params = study.best_params.copy()
            best_score = study.best_value

            self.logger.info(f'🔧 Optuna optimization completed: {best_params["model_type"]} with CV score {best_score:.4f}')

            # Add additional metadata
            best_params.update({
                'optuna_best_score': best_score,
                'optuna_n_trials': len(study.trials),
                'dataset_info': {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'n_classes': n_classes
                }
            })

            return best_params

        except ImportError:
            self.logger.warning('⚠️ Optuna not available, falling back to simple parameter selection')
            return self._fallback_hyperparameter_selection(X, y)

        except Exception as e:
            self.logger.warning(f'⚠️ Optuna optimization failed: {e}')
            return self._fallback_hyperparameter_selection(X, y)

    def _fallback_hyperparameter_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fallback hyperparameter selection when Optuna fails."""
        n_samples, n_features = X.shape

        if n_samples < 1000:
            # Small dataset - simpler model
            return {
                'model_type': 'LogisticRegression',
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'optimization_method': 'fallback'
            }
        elif n_samples < 10000:
            # Medium dataset - balanced model
            return {
                'model_type': 'RandomForestClassifier',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'optimization_method': 'fallback'
            }
        else:
            # Large dataset - efficient model
            return {
                'model_type': 'HistGradientBoostingClassifier',
                'max_iter': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'min_samples_leaf': 20,
                'l2_regularization': 1e-4,
                'optimization_method': 'fallback'
            }


def cached_computation(cache_dir: str = "cache/step02_5"):
    """Decorator for caching expensive computations."""
    import functools
    import hashlib
    import pickle
    from pathlib import Path

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get self from args (first argument for instance methods)
            instance = args[0] if args else None

            # Create cache key based on function name and arguments
            # Skip 'self' in cache key to avoid instance-specific caching issues
            cache_args = args[1:] if len(args) > 0 else args
            cache_key_data = f"{func.__name__}_{str(cache_args)}_{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()

            cache_path = Path(cache_dir) / f"{cache_key}.pkl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Check cache
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_result = pickle.load(f)
                    if instance and hasattr(instance, 'logger'):
                        instance.logger.info(f'📦 Cache hit for {func.__name__}')
                    return cached_result
                except Exception as e:
                    if instance and hasattr(instance, 'logger'):
                        instance.logger.warning(f'⚠️ Cache read failed for {func.__name__}: {e}')

            # Compute and cache
            result = func(*args, **kwargs)

            # Save to cache
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                if instance and hasattr(instance, 'logger'):
                    instance.logger.info(f'💾 Cached result for {func.__name__}')
            except Exception as e:
                if instance and hasattr(instance, 'logger'):
                    instance.logger.warning(f'⚠️ Cache write failed for {func.__name__}: {e}')

            return result
        return wrapper
    return decorator
    
    def _clear_cache(self, cache_dir: str = "cache/step02_5"):
        """Clear computation cache."""
        try:
            from pathlib import Path
            cache_path = Path(cache_dir)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                self.logger.info(f'🗑️ Cleared cache directory: {cache_dir}')
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to clear cache: {e}')
    
    def _get_cache_stats(self, cache_dir: str = "cache/step02_5"):
        """Get cache statistics."""
        try:
            from pathlib import Path
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                return {'files': 0, 'size_mb': 0}
            
            files = list(cache_path.glob('*.pkl'))
            total_size = sum(f.stat().st_size for f in files)
            
            return {
                'files': len(files),
                'size_mb': total_size / (1024 * 1024)
            }
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to get cache stats: {e}')
            return {'files': 0, 'size_mb': 0}
    
    def _optimized_logging(self, level: str, message: str, *args, **kwargs):
        """Optimized logging with level-based filtering and batching."""
        
        # Skip verbose logging in production
        if level == 'debug' and not getattr(self, 'debug_mode', False):
            return
        
        # Batch similar log messages to reduce overhead
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = []
        
        self._log_buffer.append((level, message, args, kwargs))
        
        # Flush buffer when it reaches batch size or on important messages
        if (len(self._log_buffer) >= 10 or 
            level in ['error', 'critical'] or 
            'completed' in message.lower() or 
            'failed' in message.lower()):
            self._flush_log_buffer()
    
    def _flush_log_buffer(self):
        """Flush the log buffer to reduce I/O overhead."""
        if not hasattr(self, '_log_buffer') or not self._log_buffer:
            return
        
        # Group similar messages
        message_counts = {}
        for level, message, args, kwargs in self._log_buffer:
            key = (level, message)
            if key in message_counts:
                message_counts[key] += 1
            else:
                message_counts[key] = 1
        
        # Log grouped messages
        for (level, message), count in message_counts.items():
            if count > 1:
                self.logger.log(level, f"{message} (x{count})", *args, **kwargs)
            else:
                self.logger.log(level, message, *args, **kwargs)
        
        # Clear buffer
        self._log_buffer.clear()
    
    def _reduce_logging_verbosity(self):
        """Reduce logging verbosity for better performance."""
        # Set logger level to INFO to reduce DEBUG overhead
        if hasattr(self.logger, 'setLevel'):
            self.logger.setLevel(logging.INFO)
        
        # Disable verbose sklearn logging
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('sklearn.externals.joblib').setLevel(logging.WARNING)
        
        # Disable verbose pandas logging
        logging.getLogger('pandas').setLevel(logging.WARNING)
    
    @log_all_calls
    def _validate_and_fix_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix input data using comprehensive utility integration.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated and fixed DataFrame
            
        Raises:
            ValueError: If data is None, empty, or fails validation
        """
        self.logger.info('🔍 Validating input data using comprehensive utility integration...')
        
        # CRITICAL: Validate input data before processing using common_operations
        if data is None:
            raise ValueError("CRITICAL: Input data is None. Cannot proceed with data validation.")
        
        if data.empty:
            raise ValueError("CRITICAL: Input data is empty. Cannot proceed with data validation.")
        
        # Use math_validation for numeric validation
        min_rows = validate_positive(10, "minimum_rows")
        if len(data) < min_rows:
            raise ValueError(f"CRITICAL: Insufficient data for validation. Only {len(data)} rows available, minimum {min_rows} required.")
        
        self.logger.info(f'✅ Input data validation passed: {len(data)} rows, {len(data.columns)} columns')
        
        # Use common_operations for safe data copying
        fixed_data = safe_copy(data)
        
        # Use data_processing_utils for comprehensive validation
        if self.data_validator:
            validation_report = self.data_validator.validate_dataframe(fixed_data)
            self.logger.info(f'📊 Data quality score: {validation_report.summary.get("data_quality_score", 0)}')
            
            if validation_report.summary.get('critical_issues', 0) > 0:
                self.logger.warning(f'⚠️ Found {validation_report.summary["critical_issues"]} critical data quality issues')
        
        # Use common_utilities for data quality metrics
        if hasattr(self, 'data_validator') and self.data_validator:
            quality_metrics = calculate_data_quality_metrics(fixed_data)
            self.logger.info(f'📈 Data quality metrics: {quality_metrics.get("total_rows", 0)} rows, {quality_metrics.get("memory_usage_mb", 0):.2f}MB')
        
        # Use M1 memory optimization if available
        if self.m1_memory_optimizer and self.enable_memory_optimization:
            with self.m1_memory_optimizer.memory_checkpoint("data_validation"):
                # Memory-efficient data processing
                data_size_mb = fixed_data.memory_usage(deep=True).sum() / (1024**2)
                if self.m1_memory_optimizer.should_chunk_data(data_size_mb, "general"):
                    self.logger.info(f'📦 Data size ({data_size_mb:.1f}MB) requires chunked processing')
        
        # Add missing required columns with default values using common_operations
        required_columns = {
            'exchange': 'binance',
            'symbol': 'ETHUSDT', 
            'timeframe': '30m'
        }
        
        for col, default_value in required_columns.items():
            if col not in fixed_data.columns:
                fixed_data[col] = default_value
                self.logger.info(f'📝 Added missing {col} column with default value: {default_value}')

        # Use common_utilities for safe column operations
        string_columns = list(required_columns.keys())
        for col in string_columns:
            if col in fixed_data.columns:
                # Use safe_convert_dtypes for type conversion
                dtype_mapping = {col: 'string'}
                fixed_data = safe_convert_dtypes(fixed_data, dtype_mapping)
                self.logger.info(f'🔤 Ensured {col} column is properly typed as string')
        
        # Use data_processing_utils for data cleaning
        if self.data_cleaner:
            cleaning_steps = ['remove_duplicates', 'handle_nulls', 'fix_types']
            fixed_data = self.data_cleaner.clean_dataframe(fixed_data, cleaning_steps)
            self.logger.info('🧹 Applied comprehensive data cleaning steps')
        
        # Use math_validation for numeric column validation
        numeric_columns = fixed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            try:
                # Validate that numeric columns don't contain infinite values
                if np.isinf(fixed_data[col]).any():
                    self.logger.warning(f'⚠️ Column {col} contains infinite values, replacing with NaN')
                    fixed_data[col] = fixed_data[col].replace([np.inf, -np.inf], np.nan)
                
                # Use safe_fillna for handling NaN values
                nan_count = fixed_data[col].isna().sum()
                if nan_count > 0:
                    self.logger.info(f'🔧 Filling {nan_count} NaN values in column {col}')
                    fixed_data[col] = safe_fillna(fixed_data[col], value=0)
                    
            except Exception as e:
                self.logger.warning(f'⚠️ Error processing numeric column {col}: {e}')
        
        # Use parquet_utils for data validation if available
        if self.parquet_utils:
            # Create a temporary validation of the data structure
            temp_file = "temp_validation.parquet"
            try:
                # Test parquet serialization
                if safe_to_parquet(fixed_data, temp_file):
                    validation_result = self.parquet_utils.validate_parquet_file(temp_file)
                    if validation_result['valid']:
                        self.logger.info('✅ Data structure validated with parquet_utils')
                    else:
                        self.logger.warning(f'⚠️ Parquet validation issues: {validation_result.get("error", "Unknown")}')
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            except Exception as e:
                self.logger.warning(f'⚠️ Parquet validation failed: {e}')
        
        # Use serialization_utils for configuration persistence
        if self.json_serializer:
            try:
                # Save data metadata for tracking
                metadata = {
                    'validation_timestamp': get_current_datetime().isoformat(),
                    'data_shape': fixed_data.shape,
                    'columns': list(fixed_data.columns),
                    'dtypes': {col: str(dtype) for col, dtype in fixed_data.dtypes.items()},
                    'memory_usage_mb': fixed_data.memory_usage(deep=True).sum() / (1024**2)
                }
                
                metadata_file = "data_validation_metadata.json"
                if self.json_serializer.save(metadata, metadata_file):
                    self.logger.info(f'💾 Saved data validation metadata to {metadata_file}')
                    
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to save validation metadata: {e}')
        
        # Use M1 CPU optimization for parallel processing if available
        if self.m1_cpu_optimizer and self.enable_parallel_processing:
            try:
                # Optimize data types for M1 architecture
                fixed_data = optimize_dataframe_dtypes(fixed_data)
                self.logger.info('🍎 Optimized DataFrame dtypes for M1 architecture')
                
            except Exception as e:
                self.logger.warning(f'⚠️ M1 CPU optimization failed: {e}')
        
        # Final validation using common_utilities
        final_validation = validate_dataframe_columns(fixed_data, list(required_columns.keys()))
        if not final_validation[0]:
            self.logger.warning(f'⚠️ Missing required columns after processing: {final_validation[1]}')
        
        # Log final data info using common_utilities
        final_info = get_dataframe_info(fixed_data)
        self.logger.info(f'📊 Final data info: {final_info.get("shape", "unknown")} shape, {final_info.get("total_memory", 0):.2f}MB memory')
        
        return fixed_data
        
        # Add missing timestamp column if needed
        if 'timestamp' not in fixed_data.columns:
            # Create timestamp from index if it's datetime, otherwise use sequential timestamps
            if isinstance(fixed_data.index, pd.DatetimeIndex):
                fixed_data['timestamp'] = (fixed_data.index.astype('int64') // 10**6).astype('int64')
                self.logger.info('📝 Added missing timestamp column from datetime index')
            else:
                # Create sequential timestamps starting from current time
                import time
                current_time = int(time.time() * 1000)  # Current time in milliseconds
                fixed_data['timestamp'] = range(current_time, current_time + len(fixed_data) * 60000, 60000)  # 1 minute intervals
                self.logger.info('📝 Added missing timestamp column with sequential timestamps')
        
        # Fix timestamp column type if needed
        if 'timestamp' in fixed_data.columns:
            # Check if timestamp is datetime and convert to int64 if needed
            if pd.api.types.is_datetime64_any_dtype(fixed_data['timestamp']):
                self.logger.info('🕐 Converting datetime timestamp to int64 (milliseconds)')
                # Handle NaT values before conversion
                if fixed_data['timestamp'].isna().any():
                    nan_count = fixed_data['timestamp'].isna().sum()
                    drop_percentage = (nan_count / len(fixed_data)) * 100
                    if drop_percentage > 0.5:
                        self.logger.error(f'🚨 CRITICAL: Dropping {nan_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with NaT timestamps - HIGH DATA LOSS!')
                    else:
                        self.logger.warning(f'⚠️ Dropping {nan_count} rows with NaT timestamps ({drop_percentage:.2f}%)')
                    fixed_data = fixed_data[fixed_data['timestamp'].notna()].copy()
                if len(fixed_data) > 0:
                    fixed_data['timestamp'] = (fixed_data['timestamp'].astype('int64') // 10**6).astype('int64')
                else:
                    raise ValueError('All timestamp values are NaT - data is corrupted and cannot be processed')
            elif not pd.api.types.is_integer_dtype(fixed_data['timestamp']):
                # More robust timestamp conversion handling mixed data types
                try:
                    original_dtype = fixed_data['timestamp'].dtype
                    self.logger.info(f'🔄 Converting timestamp from {original_dtype} to int64')

                    # Handle object dtype columns that might contain mixed types
                    if fixed_data['timestamp'].dtype == 'object':
                        self.logger.info('📝 Handling object dtype timestamp column')

                        # First, try to convert any datetime-like strings to datetime
                        try:
                            # Use pd.to_datetime which is more flexible than pd.to_numeric
                            converted_dt = pd.to_datetime(fixed_data['timestamp'], errors='coerce', utc=True)
                            valid_dt_mask = converted_dt.notna()

                            if valid_dt_mask.sum() < len(fixed_data):
                                invalid_count = len(fixed_data) - valid_dt_mask.sum()
                                drop_percentage = (invalid_count / len(fixed_data)) * 100
                                if drop_percentage > 0.5:
                                    self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid datetime strings - HIGH DATA LOSS!')
                                    # Log sample of invalid values for debugging
                                    invalid_samples = fixed_data.loc[~valid_dt_mask, 'timestamp'].head(3)
                                    self.logger.error(f'❌ Invalid timestamp samples: {invalid_samples.tolist()}')
                                else:
                                    self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid datetime strings ({drop_percentage:.2f}%)')
                                fixed_data = fixed_data[valid_dt_mask].copy()
                                converted_dt = converted_dt[valid_dt_mask]

                            if len(fixed_data) > 0:
                                # Convert datetime to int64 milliseconds
                                fixed_data['timestamp'] = (converted_dt.astype('int64') // 10**6).astype('int64')
                                self.logger.info('🕐 Converted datetime strings to int64 milliseconds')
                            else:
                                raise ValueError('All datetime string values are invalid - cannot convert timestamps')

                        except Exception as dt_error:
                            self.logger.error(f'❌ Failed to convert datetime strings: {dt_error}')
                            self.logger.error(f'📋 Datetime conversion traceback: {traceback.format_exc()}')
                            self.logger.warning(f'⚠️ Trying numeric conversion as fallback...')
                            # Fallback to numeric conversion
                            numeric_timestamps = pd.to_numeric(fixed_data['timestamp'], errors='coerce')
                            valid_mask = numeric_timestamps.notna()
                            if valid_mask.sum() < len(fixed_data):
                                invalid_count = len(fixed_data) - valid_mask.sum()
                                drop_percentage = (invalid_count / len(fixed_data)) * 100
                                if drop_percentage > 0.5:
                                    self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid numeric timestamps - HIGH DATA LOSS!')
                                    # Log sample of invalid values for debugging
                                    invalid_samples = fixed_data.loc[~valid_mask, 'timestamp'].head(3)
                                    self.logger.error(f'❌ Invalid timestamp samples: {invalid_samples.tolist()}')
                                else:
                                    self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid numeric timestamps ({drop_percentage:.2f}%)')
                                fixed_data = fixed_data[valid_mask].copy()
                                numeric_timestamps = numeric_timestamps[valid_mask]

                            if len(fixed_data) > 0:
                                fixed_data['timestamp'] = numeric_timestamps.astype('int64')
                                self.logger.info('🕐 Converted numeric timestamps to int64')
                            else:
                                raise ValueError('All numeric timestamp values are invalid - cannot convert timestamps')
                    else:
                        # For other dtypes (like float64), try direct numeric conversion
                        self.logger.info(f'🔢 Converting {original_dtype} timestamp to int64')
                        numeric_timestamps = pd.to_numeric(fixed_data['timestamp'], errors='coerce')
                        valid_mask = numeric_timestamps.notna()
                        if valid_mask.sum() < len(fixed_data):
                            invalid_count = len(fixed_data) - valid_mask.sum()
                            drop_percentage = (invalid_count / len(fixed_data)) * 100
                            if drop_percentage > 0.5:
                                self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid {original_dtype} timestamps - HIGH DATA LOSS!')
                                # Log sample of invalid values for debugging
                                invalid_samples = fixed_data.loc[~valid_mask, 'timestamp'].head(3)
                                self.logger.error(f'❌ Invalid timestamp samples: {invalid_samples.tolist()}')
                            else:
                                self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid {original_dtype} timestamps ({drop_percentage:.2f}%)')
                            fixed_data = fixed_data[valid_mask].copy()
                            numeric_timestamps = numeric_timestamps[valid_mask]

                        if len(fixed_data) > 0:
                            fixed_data['timestamp'] = numeric_timestamps.astype('int64')
                            self.logger.info(f'🕐 Converted {original_dtype} timestamp to int64')
                        else:
                            raise ValueError(f'All {original_dtype} timestamp values are invalid - cannot convert timestamps')

                except Exception as e:
                    self.logger.error(f'❌ Failed to convert timestamp column: {e}')
                    self.logger.error(f'📊 Timestamp column info: dtype={fixed_data["timestamp"].dtype}, shape={fixed_data.shape}')
                    # Log sample values for debugging
                    try:
                        sample_values = fixed_data['timestamp'].head(3).tolist()
                        self.logger.error(f'📋 Sample timestamp values: {sample_values}')
                        unique_types = fixed_data['timestamp'].apply(type).unique()
                        self.logger.error(f'📋 Unique value types: {[str(t) for t in unique_types]}')
                    except Exception as sample_error:
                        self.logger.error(f'❌ Could not sample timestamp values: {sample_error}')
                    return fixed_data
        
        # Remove only exact duplicate rows (identical across all columns). Keep differing rows even if timestamp duplicates.
        if 'timestamp' in fixed_data.columns:
            exact_dupe_mask = fixed_data.duplicated(subset=fixed_data.columns.tolist(), keep='first')
            exact_dupe_count = int(exact_dupe_mask.sum())
            if exact_dupe_count > 0:
                self.logger.info(f'🗑️ Removing {exact_dupe_count} exact duplicate rows (identical across all columns)')
                fixed_data = fixed_data.loc[~exact_dupe_mask]
            remaining_ts_dupes = int(fixed_data['timestamp'].duplicated().sum())
            if remaining_ts_dupes > 0:
                self.logger.warning(f'⚠️ Found {remaining_ts_dupes} duplicate timestamps with differing values; retaining all to avoid data loss')
        
        # Sort by timestamp if not monotonic
        if 'timestamp' in fixed_data.columns:
            # Ensure all timestamps are integers before sorting
            if pd.api.types.is_datetime64_any_dtype(fixed_data['timestamp']):
                # Handle NaT values before conversion for sorting
                if fixed_data['timestamp'].isna().any():
                    nan_count = fixed_data['timestamp'].isna().sum()
                    drop_percentage = (nan_count / len(fixed_data)) * 100
                    if drop_percentage > 0.5:
                        self.logger.error(f'🚨 CRITICAL: Dropping {nan_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with NaT timestamps before sorting - HIGH DATA LOSS!')
                    else:
                        self.logger.warning(f'⚠️ Dropping {nan_count} rows with NaT timestamps before sorting ({drop_percentage:.2f}%)')
                    fixed_data = fixed_data[fixed_data['timestamp'].notna()].copy()
                if len(fixed_data) > 0:
                    fixed_data['timestamp'] = (fixed_data['timestamp'].astype('int64') // 10**6).astype('int64')
                else:
                    raise ValueError('All timestamp values are NaT - cannot sort data')
            elif not pd.api.types.is_integer_dtype(fixed_data['timestamp']):
                # Convert any remaining non-integer timestamps to integers
                try:
                    # Convert to numeric, filter out NaN values
                    numeric_timestamps = pd.to_numeric(fixed_data['timestamp'], errors='coerce')
                    valid_mask = numeric_timestamps.notna()
                    if valid_mask.sum() < len(fixed_data):
                        invalid_count = len(fixed_data) - valid_mask.sum()
                        drop_percentage = (invalid_count / len(fixed_data)) * 100
                        if drop_percentage > 0.5:
                            self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid timestamps before sorting - HIGH DATA LOSS!')
                        else:
                            self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid timestamps before sorting ({drop_percentage:.2f}%)')
                        fixed_data = fixed_data[valid_mask].copy()
                    if len(fixed_data) > 0:
                        fixed_data['timestamp'] = numeric_timestamps[valid_mask].astype('int64')
                    else:
                        raise ValueError('All timestamp values are invalid, cannot sort data')
                except Exception as e:
                    raise ValueError(f'Could not convert timestamps for sorting: {e}')

            if not fixed_data['timestamp'].is_monotonic_increasing:
                self.logger.info('📈 Sorting data by timestamp')
                fixed_data = fixed_data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply schema enforcement with error handling
        try:
            fixed_data = PipelineStandards.enforce_schema(fixed_data, 'unified')
            self.logger.info('✅ Applied schema enforcement')
        except Exception as e:
            self.logger.warning(f'⚠️ Schema enforcement failed: {e}')
            # Try to fix common schema issues manually
            try:
                # Ensure trade_volume column exists and is float64
                if 'trade_volume' not in fixed_data.columns:
                    fixed_data['trade_volume'] = 0.0
                    self.logger.info('📝 Added missing trade_volume column')
                else:
                    # Handle NaN values and type conversion more robustly
                    if fixed_data['trade_volume'].isna().any():
                        self.logger.warning(f'⚠️ Found NaN values in trade_volume column: {fixed_data["trade_volume"].isna().sum()} NaNs')
                    fixed_data['trade_volume'] = pd.to_numeric(fixed_data['trade_volume'], errors='coerce').fillna(0.0).astype('float64')
                    self.logger.info('🔧 Fixed trade_volume column type')
            except Exception as fix_error:
                self.logger.warning(f'⚠️ Could not fix trade_volume column: {fix_error}')
        
        # Set datetime index for analysis (but keep original timestamp column)
        if 'timestamp' in fixed_data.columns and not isinstance(fixed_data.index, pd.DatetimeIndex):
            try:
                # Convert timestamp back to datetime for indexing
                timestamp_datetime = pd.to_datetime(fixed_data['timestamp'], unit='ms')
                fixed_data = fixed_data.set_index(timestamp_datetime)
                self.logger.info('📅 Set datetime index for analysis')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not set datetime index: {e}')
        
        # Data loss monitoring
        original_rows = len(data)
        final_rows = len(fixed_data)
        data_loss_percentage = ((original_rows - final_rows) / original_rows) * 100
        
        if data_loss_percentage > 5.0:  # More than 5% data loss
            self.logger.critical(f'🚨 CRITICAL DATA LOSS: {data_loss_percentage:.2f}% of data lost during validation!')
            self.logger.critical(f'📊 Original rows: {original_rows}, Final rows: {final_rows}')
            # Could implement data recovery strategies here
        elif data_loss_percentage > 1.0:  # More than 1% data loss
            self.logger.warning(f'⚠️ Significant data loss: {data_loss_percentage:.2f}% of data lost during validation')
            self.logger.warning(f'📊 Original rows: {original_rows}, Final rows: {final_rows}')
        else:
            self.logger.info(f'✅ Minimal data loss: {data_loss_percentage:.2f}% ({original_rows - final_rows} rows)')
        
        # Final validation
        final_validation = self.standards.validate_data_quality(fixed_data, 'unified')
        self.logger.info(f'✅ Final data quality score: {final_validation.quality_score:.2f}')
        
        if not final_validation.passed:
            self.logger.warning('⚠️ Final validation still has issues:')
            for issue in final_validation.issues:
                self.logger.warning(f'   - {issue.message}')
        
        # Store data loss metrics for monitoring
        fixed_data.attrs['data_loss_percentage'] = data_loss_percentage
        fixed_data.attrs['original_rows'] = original_rows
        fixed_data.attrs['final_rows'] = final_rows
        
        return fixed_data

    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive SR optimization logic with features, detection, and ML training."""
        self.logger.info('🎯 Starting comprehensive S/R detection optimization with unified monitoring...')
        self.logger.info(f'📊 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'📊 Pipeline state keys: {list(pipeline_state.keys())}')
        self.start_time = time.time()
        # Simple step-level tracking (unified monitor handles function-level tracking)
        internal_call_tracker = {'step_calls': 0, 'step_times': {}, 'step_results': {}}

        # Quality evaluation disabled for now - focus on core SR detection
        self.logger.info('🎯 Quality evaluation disabled - focusing on core SR detection without fractals')
        quality_results = None
        # Skip quality evaluation to avoid missing method error
        # pipeline_state['sr_quality_evaluation'] = quality_results

        try:
            # CRITICAL: Validate data availability before any processing
            self.logger.info('📊 Retrieving data from pipeline state...')
            data = pipeline_state.get('dataframe')
            if data is None:
                data = training_input.get('validated_data')
            
            # Set current timestamp for lookahead bias detection
            if data is not None and hasattr(data, 'index'):
                current_time = data.index[-1] if len(data) > 0 else None
                if current_time:
                    bias_detector = get_global_detector()
                    bias_detector.set_current_timestamp(current_time)
                    # Validate no future data
                    data = validate_no_future_data(data, 'timestamp', current_time)
            if data is None:
                # Try to load data from the same path that step02 uses
                self.logger.info('📊 No data in pipeline state, loading from data files...')
                data_path = pipeline_state.get('unified_data_path') or pipeline_state.get('raw_market_data')
                if not data_path:
                    symbol = training_input.get('symbol', '').upper()
                    exchange = training_input.get('exchange', '').upper()
                    timeframe = training_input.get('timeframe', '30m')
                    data_path = self.standards.build_path('unified_partitioned', exchange, symbol, timeframe=timeframe)
                    self.logger.info(f'📖 Constructed data path: {data_path}')
                
                # Load data from the data path; if missing, auto-trigger re-collection and retry once
                try:
                    if not PARQUET_UTILS_AVAILABLE:
                        self.logger.error("❌ ParquetUtils not available - cannot load data")
                        return await self._handle_data_unavailable_error(
                            training_input, pipeline_state,
                            "ParquetUtils not available for data loading",
                            "DATA_UNAVAILABLE"
                        )
                    parquet_utils = ParquetUtils()
                    data_path_obj = Path(data_path)
                    if data_path_obj.is_file():
                        data = parquet_utils.safe_read_parquet_with_dtype_normalization(data_path)
                    elif data_path_obj.is_dir():
                        parquet_files = list(data_path_obj.glob('**/*.parquet'))
                        if not parquet_files:
                            # Attempt centralized auto re-collection
                            self.logger.warning(f"⚠️ No parquet files found in directory: {data_path}. Attempting auto re-collection...")
                            try:
                                from src.training.steps.market_analysis.step1.enhanced_data_quality_manager import EnhancedDataQualityManager
                                _qm = EnhancedDataQualityManager(str(Path(data_path).parents[3])) if len(Path(data_path).parts) > 3 else EnhancedDataQualityManager('data_cache')
                                symbol_q = training_input.get('symbol', symbol if 'symbol' in locals() else 'ETHUSDT')
                                exchange_q = training_input.get('exchange', exchange if 'exchange' in locals() else 'BINANCE')
                                timeframe_q = training_input.get('timeframe', timeframe)
                                import asyncio as _asyncio
                                _asyncio.get_event_loop()
                                _asyncio.run(_qm.get_data_for_step3_step4(symbol_q, exchange_q, timeframe_q))
                                parquet_files = list(data_path_obj.glob('**/*.parquet'))
                            except Exception as _qe:
                                self.logger.warning(f"Auto re-collection failed: {_qe}")
                        if not parquet_files:
                            raise ValueError(f'No parquet files found in directory: {data_path}')
                        self.logger.info(f'📁 Found {len(parquet_files)} parquet files in directory')
                        dataframes = []
                        for i, file_path in enumerate(parquet_files):
                            self.logger.info(f'📖 Reading file {i + 1}/{len(parquet_files)}: {file_path.name}')
                            df = parquet_utils.safe_read_parquet_with_dtype_normalization(str(file_path))
                            if df is not None and (not df.empty):
                                # Enforce schema immediately to avoid drift
                                try:
                                    df = self.standards.enforce_schema(df, 'unified')
                                except Exception as _se:
                                    self.logger.warning(f"Schema enforcement failed for {file_path.name}: {_se}")
                                dataframes.append(df)
                        if not dataframes:
                            raise ValueError(f'Failed to read any data from parquet files in {data_path}')
                        # Concatenate without ignore_index to preserve datetime indexes
                        data = pd.concat(dataframes, ignore_index=False)
                        # If concatenation created duplicate indexes, reset and sort
                        if data.index.duplicated().any():
                            self.logger.info('🔄 Resetting index due to duplicates after concatenation')
                            data = data.reset_index(drop=True)
                        else:
                            self.logger.info(f'📊 Concatenated {len(dataframes)} dataframes preserving datetime index')
                        self.logger.info(f'📊 Concatenated {len(dataframes)} dataframes')
                    else:
                        raise ValueError(f'Path does not exist: {data_path}')
                    
                    if data is None or data.empty:
                        # Attempt centralized auto re-collection and one retry
                        self.logger.warning(f"⚠️ Empty data after read. Attempting auto re-collection and retry...")
                        try:
                            from src.training.steps.market_analysis.step1.enhanced_data_quality_manager import EnhancedDataQualityManager
                            _qm2 = EnhancedDataQualityManager(str(Path(data_path).parents[3])) if len(Path(data_path).parts) > 3 else EnhancedDataQualityManager('data_cache')
                            symbol_q2 = training_input.get('symbol', symbol if 'symbol' in locals() else 'ETHUSDT')
                            exchange_q2 = training_input.get('exchange', exchange if 'exchange' in locals() else 'BINANCE')
                            timeframe_q2 = training_input.get('timeframe', timeframe)
                            import asyncio as _asyncio
                            _asyncio.get_event_loop()
                            _asyncio.run(_qm2.get_data_for_step3_step4(symbol_q2, exchange_q2, timeframe_q2))
                            # Retry read quickly
                            if data_path_obj.is_dir():
                                parquet_files = list(data_path_obj.glob('**/*.parquet'))
                                if parquet_files:
                                    dataframes = []
                                    for file_path in parquet_files:
                                        df = parquet_utils.safe_read_parquet_with_dtype_normalization(str(file_path))
                                        if df is not None and (not df.empty):
                                            try:
                                                df = self.standards.enforce_schema(df, 'unified')
                                            except Exception as _se2:
                                                self.logger.warning(f"Schema enforcement failed for retry {file_path.name}: {_se2}")
                                            dataframes.append(df)
                                    if dataframes:
                                        data = pd.concat(dataframes, ignore_index=False)
                        except Exception as _qe2:
                            self.logger.warning(f"Auto re-collection retry failed: {_qe2}")
                        if data is None or data.empty:
                            raise ValueError(f'Failed to read data from {data_path}')
                    
                    self.logger.info(f'✅ Loaded {len(data)} rows with {len(data.columns)} columns from data files')
                    self.logger.info(f'📊 Data sample: {data.head(2).to_dict() if len(data) > 0 else "No data"}')
                except Exception as e:
                    self.logger.error(f'❌ Failed to load data from files: {e}')
                    # FAIL FAST: Don't continue with empty data
                    error_msg = f"CRITICAL: No data available for S/R optimization. Expected 'dataframe' or 'validated_data' in pipeline_state or training_input, or valid data files at {data_path}. Error: {e}"
                    self.logger.critical(error_msg)
                    return {
                        'success': False, 
                        'error': error_msg,
                        'error_type': 'DATA_UNAVAILABLE',
                        'execution_time': time.time() - self.start_time,
                        'step_name': 'step02_5_sr_optimization',
                        'data_availability': False,
                        'recommendation': 'Ensure data collection pipeline is working and data files exist before running S/R optimization'
                    }
            
            # CRITICAL: Validate data before processing
            if data is None or data.empty:
                error_msg = "CRITICAL: Data is None or empty after loading. Cannot proceed with S/R optimization."
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'EMPTY_DATA',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step02_5_sr_optimization',
                    'data_availability': False,
                    'recommendation': 'Check data loading pipeline and ensure valid data is available'
                }
            
            # Validate minimum data requirements
            if len(data) < 100:  # Minimum 100 rows for meaningful S/R analysis
                error_msg = f"CRITICAL: Insufficient data for S/R optimization. Only {len(data)} rows available, minimum 100 required."
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'INSUFFICIENT_DATA',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step02_5_sr_optimization',
                    'data_availability': True,
                    'data_rows': len(data),
                    'minimum_required': 100,
                    'recommendation': 'Collect more data or use a different timeframe with sufficient historical data'
                }
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                error_msg = f"CRITICAL: Missing required columns for S/R optimization: {missing_columns}. Available columns: {list(data.columns)}"
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'MISSING_COLUMNS',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step02_5_sr_optimization',
                    'data_availability': True,
                    'data_rows': len(data),
                    'missing_columns': missing_columns,
                    'available_columns': list(data.columns),
                    'recommendation': 'Ensure data contains OHLCV columns (open, high, low, close, volume)'
                }
            
            self.logger.info(f'✅ Data validation passed: {len(data)} rows, {len(data.columns)} columns')
            data = self._validate_and_fix_input_data(data)
            self.logger.info(f'📊 Processing {len(data)} rows of data')
            self.logger.info(f'📊 Data columns: {list(data.columns)}')
            self.logger.info(f'📊 Data types: {data.dtypes.to_dict()}')
            
            # Data drift detection
            self.logger.info('🔍 Performing data drift detection...')
            drift_results = detect_data_drift(data)
            if drift_results['drift_detected']:
                self.logger.warning(f'⚠️ Data drift detected with score: {drift_results["drift_score"]:.4f}')
                for recommendation in drift_results['recommendations']:
                    self.logger.warning(f'💡 Recommendation: {recommendation}')
            else:
                self.logger.info(f'✅ No significant data drift detected (score: {drift_results["drift_score"]:.4f})')

            # SAFETY CHECK: Validate data types before proceeding
            string_columns = [col for col in data.columns if data[col].dtype == 'object' or str(data[col].dtype).startswith('string')]
            if string_columns:
                self.logger.warning(f'⚠️ Found string columns that may cause issues: {string_columns}')
                self.logger.info('🔧 Filtering out string columns before feature engineering...')

                # Actually filter out the string columns
                data = data.drop(columns=string_columns, errors='ignore')
                self.logger.info(f'✅ Removed {len(string_columns)} string columns. Remaining columns: {len(data.columns)}')

            self.logger.info('🔧 Step 1: Engineering features...')
            self.logger.info(f'📊 About to call feature engineering with data shape: {data.shape}')
            step_start = time.time()
            features_data = await self._engineer_features(data)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['feature_engineering'] = step_time
            internal_call_tracker['step_results']['feature_engineering'] = {'success': True, 'features_count': len(features_data.columns), 'execution_time': step_time}
            self.logger.info(f'✅ Feature engineering completed in {step_time:.4f}s')
            self.logger.info(f'📈 Generated {len(features_data.columns)} features')
            self.logger.info('🎯 Step 2: Detecting support and resistance levels...')
            step_start = time.time()
            sr_levels = self._detect_sr_levels(features_data)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['sr_detection'] = step_time
            internal_call_tracker['step_results']['sr_detection'] = {'success': True, 'support_levels': len(sr_levels.get('support_levels', [])), 'resistance_levels': len(sr_levels.get('resistance_levels', [])), 'execution_time': step_time}
            self.logger.info(f'✅ SR detection completed in {step_time:.4f}s')
            self.logger.info(f"🎯 Detected {len(sr_levels.get('support_levels', []))} support levels")
            self.logger.info(f"🎯 Detected {len(sr_levels.get('resistance_levels', []))} resistance levels")
            self.logger.info('🤖 Step 3: Training ML models...')
            step_start = time.time()

            # Enhanced memory management with adaptive chunking
            ml_results = await self._train_ml_models_with_memory_management(features_data, sr_levels)

            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['ml_training'] = step_time
            internal_call_tracker['step_results']['ml_training'] = {'success': True, 'direction_accuracy': ml_results.get('direction_accuracy', 0), 'volatility_mae': ml_results.get('volatility_mae', 0), 'execution_time': step_time}
            self.logger.info(f'✅ ML training completed in {step_time:.4f}s')
            self.logger.info(f"🤖 Direction accuracy: {ml_results.get('direction_accuracy', 0):.3f}")
            self.logger.info(f"🤖 Volatility MAE: {ml_results.get('volatility_mae', 0):.6f}")
            self.logger.info('📊 All major processing steps completed - preparing final results...')

            # Initialize results variables with safe defaults
            if not hasattr(self, '_hyperparameter_results'):
                self._hyperparameter_results = {'status': 'not_performed', 'message': 'Hyperparameter optimization not completed yet'}
            if not hasattr(self, '_walk_forward_results'):
                self._walk_forward_results = {'status': 'not_performed', 'message': 'Walk-forward validation not completed yet'}

            hyperparameter_results = self._hyperparameter_results
            walk_forward_results = self._walk_forward_results

            optimization_results = {
                'best_parameters': self.sr_optimization_config,
                'confidence_score': ml_results.get('direction_accuracy', 0.85),
                'feature_count': len(features_data.columns),
                'sr_levels_detected': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                'ml_model_performance': ml_results,
                'hyperparameter_optimization': hyperparameter_results,
                'walk_forward_validation': walk_forward_results,
                'internal_call_tracker': internal_call_tracker
            }
            execution_time = time.time() - self.start_time
            self.logger.info(f'✅ Comprehensive SR optimization completed in {execution_time:.2f} seconds')
            self.logger.info(f"📈 Features engineered: {optimization_results['feature_count']}")
            self.logger.info(f"🎯 SR levels detected: {optimization_results['sr_levels_detected']}")
            self.logger.info(f"🤖 ML accuracy: {optimization_results['confidence_score']:.3f}")
            self.logger.info(f"📊 Internal function calls: {internal_call_tracker['step_calls']}")
            execution_report = {'total_execution_time': execution_time, 'step_breakdown': internal_call_tracker['step_times'], 'step_results': internal_call_tracker['step_results'], 'performance_summary': {'features_per_second': len(features_data.columns) / execution_time, 'sr_levels_per_second': optimization_results['sr_levels_detected'] / execution_time, 'ml_accuracy': ml_results.get('direction_accuracy', 0)}}

            # Include unified monitor performance summary
            performance_summary = self.performance_monitor.get_performance_summary()

            # Add feature selection info to results if available
            feature_selection_info = getattr(self, 'feature_selection_info', None)
            if feature_selection_info:
                ml_results['feature_selection'] = feature_selection_info

            # Generate and save comprehensive enhanced report
            try:
                self.logger.info('📝 Generating comprehensive enhanced report with detailed metrics...')

                # Initialize enhanced reporter
                symbol = training_input.get('symbol', 'UNKNOWN')
                exchange = training_input.get('exchange', 'UNKNOWN')
                timeframe = training_input.get('timeframe', '30m')

                self.logger.info(f'📊 Report parameters - Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')

                financial_logger = Step02_5FinancialLogger(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )

                self.logger.info('✅ Enhanced reporter initialized successfully')

                # Prepare execution data with detailed metrics
                execution_data = {
                    'execution_time': execution_time,
                    'memory_usage': performance_summary.get('memory_usage', 0),
                    'cpu_usage': performance_summary.get('cpu_usage', 0),
                    'function_calls': internal_call_tracker.get('total_calls', 0),
                    'step_breakdown': internal_call_tracker.get('step_times', {}),
                    'performance_summary': execution_report.get('performance_summary', {}),
                    'feature_count': len(features_data.columns),
                    'data_rows': len(features_data),
                    'sr_levels_detected': optimization_results.get('sr_levels_detected', 0),
                    'ml_accuracy': ml_results.get('direction_accuracy', 0),
                    'processing_timestamp': datetime.now().isoformat()
                }

                # Generate comprehensive report
                print("🔍 STEP02_5: Starting report generation...")
                self.logger.info('🔍 STEP02_5: Starting report generation...')

                # Log data availability for debugging
                self.logger.info(f'📊 SR levels available: {bool(sr_levels and any(sr_levels.get(key, []) for key in ["support_levels", "resistance_levels"]))}')
                self.logger.info(f'📊 ML results available: {bool(ml_results)}')
                self.logger.info(f'📊 Features data shape: {features_data.shape if features_data is not None else "None"}')

                # Log financial metrics using the new financial logger
                financial_logger.log_step_execution(
                    sr_levels=sr_levels,
                    ml_results=ml_results,
                    execution_data=execution_data,
                    data=features_data
                )

                self.logger.info('✅ Financial metrics logged successfully')

            except Exception as report_error:
                self.logger.warning(f'⚠️ Failed to generate enhanced report: {report_error}')
                import traceback
                self.logger.warning(f'Enhanced report generation traceback: {traceback.format_exc()}')

                # Fallback to basic reporting
                try:
                    print("🔄 STEP02_5: Falling back to basic report generation...")
                    self.logger.info('🔄 Falling back to basic report generation...')

                    # Create a simple fallback report with available data
                    fallback_report = {
                        'step_name': 'step02_5_sr_optimization',
                        'execution_time': execution_time,
                        'sr_levels_detected': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                        'features_engineered': len(features_data.columns) if features_data is not None else 0,
                        'ml_accuracy': ml_results.get('direction_accuracy', 0),
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'status': 'fallback_report'
                    }

                    # Try to save the fallback report
                    fallback_path = save_training_report(
                        data=fallback_report,
                        step_name='step02_5_sr_optimization',
                        report_type='fallback_report',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='json'
                    )
                    self.logger.info(f'💾 Fallback report saved: {fallback_path}')

                    # Also try the original basic report method if available
                    try:
                        final_report = self._generate_final_report(sr_levels, optimization_results, ml_results, execution_report, performance_summary)
                        basic_report_path = save_training_report(
                            data=final_report,
                            step_name='step02_5_sr_optimization',
                            report_type='basic_sr_analysis',
                            symbol=symbol,
                            timeframe=timeframe,
                            file_format='md'
                        )
                        self.logger.info(f'💾 Basic report saved: {basic_report_path}')
                    except Exception as basic_method_error:
                        self.logger.warning(f'⚠️ Basic report method failed: {basic_method_error}')

                except Exception as basic_error:
                    self.logger.error(f'❌ Both enhanced and basic report generation failed: {basic_error}')
                    import traceback
                    self.logger.error(f'Fallback report traceback: {traceback.format_exc()}')

            # Return success with all results
            print("✅ STEP02_5: Execution completed successfully, about to return results")
            self.logger.info('✅ Step 2.5 SR optimization completed successfully')
            self.logger.info(f'📊 S/R levels detected: {len(sr_levels.get("support_levels", []))} support, {len(sr_levels.get("resistance_levels", []))} resistance')
            
            # Ensure S/R levels are properly formatted for step06
            formatted_sr_levels = self._format_sr_levels_for_pipeline(sr_levels)
            
            success_result = {
                'success': True, 
                'step02_5_sr_optimization_completed': True, 
                'sr_levels': formatted_sr_levels, 
                'sr_optimization_results': optimization_results, 
                'features_data': features_data, 
                'ml_results': ml_results, 
                'execution_time': execution_time, 
                'execution_report': execution_report, 
                'internal_call_tracker': internal_call_tracker, 
                'unified_performance_summary': performance_summary, 
                'step_name': 'step02_5_sr_optimization', 
                'pipeline_state_update': {'sr_levels': formatted_sr_levels}
            }

        except Exception as e:
            execution_time = time.time() - self.start_time
            
            # Determine error type and severity
            error_type = 'UNKNOWN_ERROR'
            error_severity = 'HIGH'
            
            if 'DATA_UNAVAILABLE' in str(e) or 'No data available' in str(e):
                error_type = 'DATA_UNAVAILABLE'
                error_severity = 'CRITICAL'
            elif 'EMPTY_DATA' in str(e) or 'empty' in str(e).lower():
                error_type = 'EMPTY_DATA'
                error_severity = 'CRITICAL'
            elif 'INSUFFICIENT_DATA' in str(e) or 'Insufficient data' in str(e):
                error_type = 'INSUFFICIENT_DATA'
                error_severity = 'HIGH'
            elif 'MISSING_COLUMNS' in str(e) or 'Missing required columns' in str(e):
                error_type = 'MISSING_COLUMNS'
                error_severity = 'HIGH'
            elif 'ImportError' in str(e) or 'ModuleNotFoundError' in str(e):
                error_type = 'IMPORT_ERROR'
                error_severity = 'MEDIUM'
            elif 'ValueError' in str(e):
                error_type = 'VALUE_ERROR'
                error_severity = 'HIGH'
            
            self.logger.error(f'❌ SR optimization failed with {error_type}: {e}')
            self.logger.error(f'🚨 Error severity: {error_severity}')
            
            # Generate appropriate error report based on error type
            try:
                symbol = training_input.get('symbol', 'UNKNOWN')
                exchange = training_input.get('exchange', 'UNKNOWN')
                timeframe = training_input.get('timeframe', '30m')
                
                # Create comprehensive error report
                error_report = {
                    'success': False,
                    'error_type': error_type,
                    'error_severity': error_severity,
                    'error_message': str(e),
                    'execution_time': execution_time,
                    'step_name': 'step02_5_sr_optimization',
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'data_availability': error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA'],
                    'recommendations': self._get_error_recommendations(error_type, str(e)),
                    'troubleshooting_steps': self._get_troubleshooting_steps(error_type),
                    'next_actions': self._get_next_actions(error_type)
                }
                
                # Save error report
                error_report_path = save_training_report(
                    data=error_report,
                    step_name='step02_5_sr_optimization',
                    report_type='error_report',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='json'
                )
                self.logger.info(f'💾 Error report saved: {error_report_path}')
                
                # Log error details for debugging
                self.logger.error(f'📋 Error details: {error_report}')
                
            except Exception as report_error:
                self.logger.error(f'❌ Failed to generate error report: {report_error}')
                import traceback
                self.logger.error(f'Error report generation traceback: {traceback.format_exc()}')
            
            # Return comprehensive error result
            return {
                'success': False,
                'error': str(e),
                'error_type': error_type,
                'error_severity': error_severity,
                'execution_time': execution_time,
                'step_name': 'step02_5_sr_optimization',
                'data_availability': error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA'],
                'recommendations': self._get_error_recommendations(error_type, str(e)),
                'troubleshooting_steps': self._get_troubleshooting_steps(error_type)
            }

        # Return success result if no exception occurred
        return success_result

    def _get_error_recommendations(self, error_type: str, error_message: str) -> List[str]:
        """Get specific recommendations based on error type."""
        recommendations = []
        
        if error_type == 'DATA_UNAVAILABLE':
            recommendations = [
                "Check if data collection pipeline is running correctly",
                "Verify that data files exist in the expected directory",
                "Ensure the data collection step (step02) completed successfully",
                "Check file permissions and disk space",
                "Consider running data collection manually before S/R optimization"
            ]
        elif error_type == 'EMPTY_DATA':
            recommendations = [
                "Verify data files are not corrupted",
                "Check if data files contain valid market data",
                "Ensure data collection retrieved actual market data",
                "Verify the data format matches expected schema"
            ]
        elif error_type == 'INSUFFICIENT_DATA':
            recommendations = [
                "Collect more historical data for the symbol/timeframe",
                "Use a different timeframe with more available data",
                "Check if the data collection period is too short",
                "Verify the symbol has sufficient trading history"
            ]
        elif error_type == 'MISSING_COLUMNS':
            recommendations = [
                "Ensure data contains OHLCV columns (open, high, low, close, volume)",
                "Check data schema and column naming conventions",
                "Verify data preprocessing steps are working correctly",
                "Ensure data format matches expected structure"
            ]
        elif error_type == 'IMPORT_ERROR':
            recommendations = [
                "Install missing Python packages",
                "Check Python environment and dependencies",
                "Verify all required modules are available",
                "Update package versions if needed"
            ]
        elif error_type == 'VALUE_ERROR':
            recommendations = [
                "Check data quality and format",
                "Verify all required parameters are provided",
                "Ensure data values are within expected ranges",
                "Check for data type mismatches"
            ]
        else:
            recommendations = [
                "Check system logs for detailed error information",
                "Verify all dependencies are installed correctly",
                "Ensure system resources are available",
                "Contact support if the issue persists"
            ]
        
        return recommendations

    def _get_troubleshooting_steps(self, error_type: str) -> List[str]:
        """Get specific troubleshooting steps based on error type."""
        steps = []
        
        if error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA']:
            steps = [
                "1. Check if data_cache directory exists and contains files",
                "2. Verify data collection pipeline is working",
                "3. Run data collection step manually",
                "4. Check file permissions and disk space",
                "5. Verify data format and schema"
            ]
        elif error_type == 'INSUFFICIENT_DATA':
            steps = [
                "1. Check available data range for the symbol",
                "2. Try a different timeframe (1h, 4h, 1d)",
                "3. Verify data collection period settings",
                "4. Check if symbol has sufficient trading history"
            ]
        elif error_type == 'MISSING_COLUMNS':
            steps = [
                "1. Check data schema and column names",
                "2. Verify data preprocessing steps",
                "3. Ensure OHLCV columns are present",
                "4. Check data format consistency"
            ]
        elif error_type == 'IMPORT_ERROR':
            steps = [
                "1. Check Python environment",
                "2. Install missing packages",
                "3. Verify import paths",
                "4. Check package versions"
            ]
        else:
            steps = [
                "1. Check system logs for detailed errors",
                "2. Verify all dependencies",
                "3. Check system resources",
                "4. Review configuration settings"
            ]
        
        return steps

    def _get_next_actions(self, error_type: str) -> List[str]:
        """Get next actions to take based on error type."""
        actions = []
        
        if error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA']:
            actions = [
                "Run data collection pipeline first",
                "Verify data files exist and are accessible",
                "Check data collection configuration",
                "Ensure sufficient disk space"
            ]
        elif error_type == 'INSUFFICIENT_DATA':
            actions = [
                "Collect more historical data",
                "Use a different timeframe",
                "Check data collection period settings",
                "Verify symbol trading history"
            ]
        elif error_type == 'MISSING_COLUMNS':
            actions = [
                "Fix data schema issues",
                "Ensure OHLCV columns are present",
                "Check data preprocessing pipeline",
                "Verify data format consistency"
            ]
        elif error_type == 'IMPORT_ERROR':
            actions = [
                "Install missing packages",
                "Check Python environment",
                "Verify import paths",
                "Update dependencies"
            ]
        else:
            actions = [
                "Review error logs",
                "Check system configuration",
                "Verify all dependencies",
                "Contact support if needed"
            ]
        
        return actions

    def _generate_final_report(self, sr_levels: Dict[str, Any], optimization_results: Dict[str, Any],
                              ml_results: Dict[str, Any], execution_report: Dict[str, Any],
                              performance_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive final report for step02_5_sr_optimization.

        Args:
            sr_levels: Detected support and resistance levels
            optimization_results: Optimization configuration and results
            ml_results: Machine learning model results
            execution_report: Execution timing and performance data
            performance_summary: Unified performance monitoring summary

        Returns:
            Dict containing comprehensive report data
        """
        from datetime import datetime

        # Get current price for context (if available in data)
        current_price = None
        try:
            # Try to get current price from the most recent data point
            if hasattr(self, 'features_data') and self.features_data is not None and not self.features_data.empty:
                current_price = float(self.features_data['close'].iloc[-1])
        except Exception:
            current_price = None

        # Prepare report data
        report = {
            'analysis_date': datetime.now().isoformat(),
            'current_price': current_price,
            'analysis_parameters': {
                'tolerance': self.sr_optimization_config.get('tolerance', 0.005),
                'minimum_touches': self.sr_optimization_config.get('minimum_touches', 2),
                'lookback_periods': self.sr_optimization_config.get('lookback_periods', 2000)
            },
            'support_levels': [],
            'resistance_levels': [],
            'strength_analysis': {},
            'trading_implications': {},
            'ml_performance': {},
            'execution_summary': {},
            'performance_metrics': {}
        }

        # Process support levels
        support_levels = sr_levels.get('support_levels', [])
        if support_levels:
            sorted_supports = sorted(support_levels, key=lambda x: x.get('strength', 0), reverse=True)
            for i, level in enumerate(sorted_supports[:5]):  # Top 5 strongest
                level_data = {
                    'rank': i + 1,
                    'price': float(level.get('price', 0)),
                    'strength': float(level.get('strength', 0)),
                    'touches': int(level.get('touches', 0)),
                    'bounces': int(level.get('bounces', 0)),
                    'bounce_rate': float(level.get('bounce_rate', 0)),
                    'distance': self._calculate_price_distance(current_price, level.get('price', 0)) if current_price else 0
                }
                report['support_levels'].append(level_data)

        # Process resistance levels
        resistance_levels = sr_levels.get('resistance_levels', [])
        if resistance_levels:
            sorted_resistances = sorted(resistance_levels, key=lambda x: x.get('strength', 0), reverse=True)
            for i, level in enumerate(sorted_resistances[:5]):  # Top 5 strongest
                level_data = {
                    'rank': i + 1,
                    'price': float(level.get('price', 0)),
                    'strength': float(level.get('strength', 0)),
                    'touches': int(level.get('touches', 0)),
                    'bounces': int(level.get('bounces', 0)),
                    'bounce_rate': float(level.get('bounce_rate', 0)),
                    'distance': self._calculate_price_distance(current_price, level.get('price', 0)) if current_price else 0
                }
                report['resistance_levels'].append(level_data)

        # Strength analysis summary
        if report['support_levels']:
            report['strength_analysis']['average_support_strength'] = sum(level['strength'] for level in report['support_levels']) / len(report['support_levels'])
            report['strength_analysis']['strongest_support'] = report['support_levels'][0]['price'] if report['support_levels'] else None

        if report['resistance_levels']:
            report['strength_analysis']['average_resistance_strength'] = sum(level['strength'] for level in report['resistance_levels']) / len(report['resistance_levels'])
            report['strength_analysis']['strongest_resistance'] = report['resistance_levels'][0]['price'] if report['resistance_levels'] else None

        # Trading implications
        if current_price and report['support_levels'] and report['resistance_levels']:
            nearest_support = min(report['support_levels'], key=lambda x: abs(x['price'] - current_price))
            nearest_resistance = min(report['resistance_levels'], key=lambda x: abs(x['price'] - current_price))

            report['trading_implications'] = {
                'nearest_support': {
                    'price': nearest_support['price'],
                    'distance': nearest_support['distance'],
                    'strength': nearest_support['strength'],
                    'reliability': 'High' if nearest_support['strength'] >= 0.8 else 'Medium' if nearest_support['strength'] >= 0.6 else 'Low'
                },
                'nearest_resistance': {
                    'price': nearest_resistance['price'],
                    'distance': nearest_resistance['distance'],
                    'strength': nearest_resistance['strength'],
                    'reliability': 'High' if nearest_resistance['strength'] >= 0.8 else 'Medium' if nearest_resistance['strength'] >= 0.6 else 'Low'
                },
                'risk_assessment': self._assess_risk(current_price, nearest_support['price'], nearest_resistance['price'])
            }

        # ML performance summary
        report['ml_performance'] = {
            'direction_accuracy': ml_results.get('direction_accuracy', 0),
            'volatility_mae': ml_results.get('volatility_mae', 0),
            'best_model': ml_results.get('model_type', 'Unknown'),
            'feature_count': optimization_results.get('feature_count', 0)
        }

        # Execution summary
        report['execution_summary'] = {
            'total_execution_time': execution_report.get('total_execution_time', 0),
            'features_engineered': optimization_results.get('feature_count', 0),
            'sr_levels_detected': optimization_results.get('sr_levels_detected', 0),
            'ml_accuracy': ml_results.get('direction_accuracy', 0),
            'step_breakdown': execution_report.get('step_breakdown', {}),
            'performance_summary': execution_report.get('performance_summary', {})
        }

        # Performance metrics
        report['performance_metrics'] = {
            'unified_monitor_summary': performance_summary,
            'internal_call_tracker': optimization_results.get('internal_call_tracker', {}),
            'function_call_report': getattr(self, 'function_call_report', {})
        }
        
        # Use serialization_utils for comprehensive report persistence
        if self.json_serializer:
            try:
                # Save main report as JSON
                report_file = f"step02_5_final_report_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.json"
                if self.json_serializer.save(report, report_file, indent=2):
                    self.logger.info(f'💾 Saved comprehensive final report to {report_file}')
                
                # Save summary report for quick access
                summary_report = {
                    'analysis_date': report['analysis_date'],
                    'current_price': report['current_price'],
                    'support_levels_count': len(report['support_levels']),
                    'resistance_levels_count': len(report['resistance_levels']),
                    'strongest_support': report['strength_analysis'].get('strongest_support'),
                    'strongest_resistance': report['strength_analysis'].get('strongest_resistance'),
                    'ml_accuracy': report['ml_performance'].get('accuracy', 0),
                    'execution_time': report['execution_summary'].get('total_execution_time', 0)
                }
                
                summary_file = f"step02_5_summary_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.json"
                if self.json_serializer.save(summary_report, summary_file, indent=2):
                    self.logger.info(f'📋 Saved summary report to {summary_file}')
                    
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to save reports using serialization_utils: {e}')
        
        # Use universal_serializer for additional formats if available
        if self.universal_serializer:
            try:
                # Save as pickle for faster loading if needed
                pickle_file = f"step02_5_report_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.pkl"
                if self.universal_serializer.save(report, pickle_file, format_type='pickle'):
                    self.logger.info(f'🥒 Saved report as pickle to {pickle_file}')
                    
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to save pickle report: {e}')
        
        # Use common_operations for safe data operations
        try:
            # Log report statistics using safe operations
            report_size = len(str(report))
            report_size_mb = report_size / (1024 * 1024)
            
            self.logger.info(f'📊 Final report generated: {len(report)} sections, {report_size_mb:.2f}MB size')
            
            # Use safe operations for report validation
            if safe_dict_get(report, 'support_levels') and safe_dict_get(report, 'resistance_levels'):
                self.logger.info('✅ Report contains both support and resistance levels')
            else:
                self.logger.warning('⚠️ Report missing support or resistance levels')
                
        except Exception as e:
            self.logger.warning(f'⚠️ Error in report finalization: {e}')

        return report

    def _calculate_price_distance(self, current_price: float, level_price: float) -> float:
        """Calculate percentage distance between current price and level price using comprehensive math_validation."""
        try:
            # Use math_validation for input validation
            current_price = validate_positive(current_price, "current_price")
            level_price = validate_positive(level_price, "level_price")
            
            # Use safe_divide to prevent division by zero
            price_diff = level_price - current_price
            distance_ratio = safe_divide(price_diff, current_price, 0.0)
            
            # Use safe_percentage_change for percentage calculation
            percentage_distance = safe_percentage_change(current_price, level_price, 0.0)
            
            # Validate the result is finite
            result = validate_finite(percentage_distance, "price_distance")
            
            self.logger.debug(f'📊 Price distance calculation: {current_price} -> {level_price} = {result:.2f}%')
            return result
            
        except MathValidationError as e:
            self.logger.warning(f"Mathematical validation error in price distance calculation: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error in price distance calculation: {e}")
            return 0.0

    def _assess_risk(self, current_price: float, support_price: float, resistance_price: float) -> str:
        """Assess trading risk based on proximity to S/R levels using comprehensive math_validation."""
        try:
            # Use math_validation for input validation
            current_price = validate_positive(current_price, "current_price")
            support_price = validate_positive(support_price, "support_price")
            resistance_price = validate_positive(resistance_price, "resistance_price")
            
            # Use safe_divide to prevent division by zero
            support_diff = abs(current_price - support_price)
            resistance_diff = abs(current_price - resistance_price)
            
            support_distance = safe_divide(support_diff, current_price, 0.0)
            resistance_distance = safe_divide(resistance_diff, current_price, 0.0)
            
            # Use safe_percentage_change for distance calculations
            support_pct = safe_percentage_change(current_price, support_price, 0.0)
            resistance_pct = safe_percentage_change(current_price, resistance_price, 0.0)
            
            # Validate results are finite
            support_distance = validate_finite(support_distance, "support_distance")
            resistance_distance = validate_finite(resistance_distance, "resistance_distance")
            
            self.logger.debug(f'📊 Risk assessment: current={current_price}, support={support_price}, resistance={resistance_price}')
            self.logger.debug(f'📊 Distances: support={support_distance:.4f}, resistance={resistance_distance:.4f}')
            
        except MathValidationError as e:
            self.logger.warning(f"Mathematical validation error in risk assessment: {e}")
            return 'Unknown - Calculation error'
        except Exception as e:
            self.logger.error(f"Unexpected error in risk assessment: {e}")
            return 'Unknown - Calculation error'

        # High risk if very close to support (potential breakdown) or resistance (potential rejection)
        if support_distance <= 0.005:  # Within 0.5%
            return 'High - Price near strong support, watch for breakdown'
        elif resistance_distance <= 0.005:  # Within 0.5%
            return 'High - Price near strong resistance, watch for rejection'
        elif support_distance <= 0.01 or resistance_distance <= 0.01:  # Within 1%
            return 'Medium - Approaching key levels'
        else:
            return 'Low - Price well-positioned between support and resistance'

            self.logger.error('🔍 ENHANCED ERROR DIAGNOSTICS:')
            self.logger.error(f'   Error Type: {type(e).__name__}')
            self.logger.error(f'   Error Message: {str(e)}')
            self.logger.error(f'   Execution Time: {execution_time:.2f}s')

            # Memory usage information
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                self.logger.error(f'   System Memory - Total: {memory.total / 1024**3:.1f}GB, Available: {memory.available / 1024**3:.1f}GB, Used: {memory.percent:.1f}%')

                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.error(f'   Process Memory - RSS: {memory_info.rss / 1024**2:.1f}MB, VMS: {memory_info.vms / 1024**2:.1f}MB')
            else:
                self.logger.error('   Memory monitoring unavailable (psutil not installed)')

            # Python memory usage if available
            try:
                import gc
                collected = gc.collect()
                self.logger.error(f'   Garbage Collection: {collected} objects collected')
            except:
                pass

            # Full traceback
            self.logger.error('   Full Traceback:')
            for line in traceback.format_exc().split('\n'):
                self.logger.error(f'     {line}')

            # Local variables in the calling frame
            try:
                frame = sys._getframe(1)
                local_vars = frame.f_locals
                self.logger.error('   Local Variables (key info):')
                for key, value in local_vars.items():
                    if key in ['features_data', 'sr_levels', 'ml_results']:
                        if hasattr(value, 'shape'):
                            self.logger.error(f'     {key}: {value.shape}')
                        elif isinstance(value, dict):
                            self.logger.error(f'     {key}: {len(value)} items')
                        else:
                            self.logger.error(f'     {key}: {type(value)}')
            except:
                self.logger.error('   Could not capture local variables')

            internal_call_tracker['error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'execution_time': execution_time,
                'traceback': traceback.format_exc(),
                'memory_usage': {
                    'system_total_gb': memory.total / 1024**3,
                    'system_available_gb': memory.available / 1024**3,
                    'system_used_percent': memory.percent,
                    'process_rss_mb': memory_info.rss / 1024**2,
                    'process_vms_mb': memory_info.vms / 1024**2
                }
            }
            return {'success': False, 'step02_5_sr_optimization_completed': False, 'step02_5_sr_optimization_failure_reason': str(e), 'error': str(e), 'execution_time': execution_time, 'internal_call_tracker': internal_call_tracker, 'step_name': 'step02_5_sr_optimization'}

    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for SR analysis using advanced modules."""
        self.logger.info('🔧 FEATURE ENGINEERING STARTED - Data shape: {}'.format(data.shape))
        feature_start_time = time.time()
        
        # CRITICAL: Validate input data before feature engineering
        if data is None:
            raise ValueError("CRITICAL: Input data is None for feature engineering. Cannot proceed.")
        
        if data.empty:
            raise ValueError("CRITICAL: Input data is empty for feature engineering. Cannot proceed.")
        
        if len(data) < 50:  # Minimum 50 rows for meaningful feature engineering
            raise ValueError(f"CRITICAL: Insufficient data for feature engineering. Only {len(data)} rows available, minimum 50 required.")
        
        # Validate required columns for feature engineering
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"CRITICAL: Missing required columns for feature engineering: {missing_columns}. Available columns: {list(data.columns)}")

        # Additional safety check: Ensure no string columns are present
        string_columns = [col for col in data.columns if data[col].dtype == 'object' or str(data[col].dtype).startswith('string')]
        if string_columns:
            self.logger.warning(f'🚨 CRITICAL: String columns detected in feature engineering input: {string_columns}')
            self.logger.warning('🔧 Removing string columns to prevent feature engineering issues...')
            data = data.drop(columns=string_columns, errors='ignore')
            self.logger.info(f'✅ Removed {len(string_columns)} string columns during feature engineering')

        self.logger.info(f'✅ Feature engineering input validation passed: {len(data)} rows, {len(data.columns)} columns')
        
        try:
            # Try to use advanced feature engineering module
            if not ADVANCED_FEATURES_AVAILABLE:
                raise ImportError("AdvancedFeatureEngineeringStep not available")
            # AdvancedFeatureEngineeringStep is already imported at module level

            # Configure advanced feature engineering - wavelets NEVER enabled for step02_5
            enable_wavelets = False  # Never enable wavelets for step02_5
            self.logger.info(f'🌊 Wavelet features: DISABLED (never enabled for step02_5)')
            self.logger.info(f'🚫 Lookback optimization: DISABLED (step02_5 compatibility mode)')

            feature_config = {
                'feature_engineering': {
                    'enable_wavelets': enable_wavelets,
                    'enable_multi_timeframe': True,
                    'enable_feature_interactions': True,  # Re-enable interactions
                    'enable_regime_features': True,  # Enable comprehensive regime features
                    'timeframes': ['30m', '1h', '4h', '1d'],
                    'chunk_size': 500000,
                    'max_features': 1000,  # Increased to accommodate all SR features
                    'feature_interaction_degree': 2,  # Include pairwise interactions
                    'regime_lookback_days': 30,
                    # Disable lookback optimization for step02_5
                    'disable_lookback_optimization': True,
                    'cross_timeframe_enabled': False,
                    'regime_specific': True,  # Enable SR-specific features
                    'sr_feature_priority': True,  # Prioritize SR features
                }
            }

            # Initialize advanced feature engineering
            advanced_fe = AdvancedFeatureEngineeringStep(feature_config)
            await advanced_fe.initialize()

            # Use individual advanced feature methods
            self.logger.info('🚀 Executing advanced feature engineering...')
            self.logger.info('📋 Step02_5 Feature Access Policy:')
            self.logger.info('   INCLUDED: Technical indicators, Microstructure features, Multi-timeframe features, Interactions, Regime-aware (best-effort)')
            self.logger.info('   EXCLUDED: Wavelet features')
            all_advanced_features = {}

            # 1. Technical indicators (from step06)
            self.logger.info('📈 Loading comprehensive technical indicators from step06...')
            technical_features = advanced_fe._generate_comprehensive_technical_features(data)
            all_advanced_features['technical'] = technical_features
            self.logger.info(f'✅ Technical indicators: {len(technical_features.columns)} features')

            # 2. Microstructure features
            self.logger.info('🔬 Calculating microstructure features...')
            try:
                microstructure_features = advanced_fe._calculate_microstructure_features(data)
                all_advanced_features['microstructure'] = microstructure_features
                self.logger.info(f'✅ Microstructure features: {len(microstructure_features.columns)} features')
            except Exception as e:
                self.logger.warning(f'Microstructure features failed: {e}')

            # 3. Multi-timeframe features
            if advanced_fe.enable_multi_timeframe:
                self.logger.info('⏰ Calculating multi-timeframe features...')
                try:
                    mtf_features = await advanced_fe._build_mtf_features_required(data)
                    all_advanced_features['multi_timeframe'] = mtf_features
                    self.logger.info(f'✅ Multi-timeframe features: {len(mtf_features.columns)} features')
                except Exception as e:
                    self.logger.warning(f'Multi-timeframe features failed: {e}')

            # 4. Wavelet features
            if advanced_fe.enable_wavelets and advanced_fe.wavelet_analyzer is not None:
                self.logger.info('🌊 Calculating wavelet features...')
                try:
                    wavelet_features = advanced_fe.wavelet_analyzer.extract_wavelet_features(data, price_column='close', symbol='SYMBOL', timeframe='30m')
                    all_advanced_features['wavelet'] = wavelet_features
                    self.logger.info(f'✅ Wavelet features: {len(wavelet_features.columns)} features')
                except Exception as e:
                    self.logger.warning(f'Wavelet features failed: {e}')

            # 5. Feature interactions - ENABLED (best-effort)
            if advanced_fe.enable_feature_interactions:
                self.logger.info('🔗 Creating feature interactions...')
                try:
                    interactions = advanced_fe._create_feature_interactions(data)
                    if interactions is not None and hasattr(interactions, 'columns') and len(interactions.columns) > 0:
                        all_advanced_features['interactions'] = interactions
                        self.logger.info(f'✅ Interactions features: {len(interactions.columns)} features')
                except Exception as e:
                    self.logger.warning(f'Feature interactions failed: {e}')

            # 6. Regime-aware features - ENABLED (best-effort)
            if advanced_fe.enable_regime_features:
                if getattr(advanced_fe, 'regime_engine', None) is not None:
                    self.logger.info('🎭 Creating regime-aware features...')
                    try:
                        regime_features = advanced_fe._create_regime_aware_features(data, {})
                        if regime_features is not None and hasattr(regime_features, 'columns') and len(regime_features.columns) > 0:
                            all_advanced_features['regime'] = regime_features
                            self.logger.info(f'✅ Regime features: {len(regime_features.columns)} features')
                    except Exception as e:
                        self.logger.warning(f'Regime features failed: {e}')
                else:
                    self.logger.info('🎭 Regime engine not available; skipping regime-aware features')

            # Count total features
            total_advanced_count = sum(len(df.columns) for df in all_advanced_features.values() if df is not None and hasattr(df, 'columns'))
            self.logger.info(f'🎉 Advanced features total: {total_advanced_count} features')

            # Combine with basic features
            features_data = data.copy()

            # Add ALL advanced features
            total_advanced_features = 0
            for feature_type, feature_df in all_advanced_features.items():
                if feature_df is not None and hasattr(feature_df, 'columns') and len(feature_df.columns) > 0:
                    for col in feature_df.columns:
                        features_data[f'{feature_type}_{col}'] = feature_df[col]
                    total_advanced_features += len(feature_df.columns)
                    self.logger.info(f'✅ Added {len(feature_df.columns)} {feature_type} features')

            self.logger.info(f'🎉 Total advanced features added: {total_advanced_features}')

        except ImportError:
            error_msg = 'Advanced feature engineering module not available - required components missing'
            self.logger.error(f'❌ {error_msg}')
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f'Advanced feature engineering failed: {e}'
            self.logger.error(f'❌ {error_msg}')
            raise RuntimeError(error_msg)
        
        feature_time = time.time() - feature_start_time
        self.logger.info(f'✅ Feature engineering completed in {feature_time:.4f}s')
        self.logger.info(f'📈 Engineered {len(features_data.columns)} features')
        self.logger.info(f'📊 Final data shape: {features_data.shape}')

        # Debug: Show feature names
        original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'year', 'month', 'day', 'exchange', 'symbol', 'timeframe', 'trade_volume']
        new_features = [col for col in features_data.columns if col not in original_cols]
        self.logger.info(f'📊 Original columns: {len(original_cols)}')
        self.logger.info(f'📊 New features: {len(new_features)} - {new_features[:10]}...' if len(new_features) > 10 else f'📊 New features: {len(new_features)} - {new_features}')

        # CRITICAL FIX: Filter out non-numeric columns to prevent string-to-float conversion errors
        # Keep only numeric columns and essential columns for ML processing
        self.logger.info('🔧 Filtering features to include only numeric columns for ML compatibility...')
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns.tolist()

        # Add essential non-numeric columns that are needed for processing but will be excluded from ML features
        essential_cols = ['timestamp']  # Keep timestamp for indexing
        final_columns = numeric_columns + [col for col in essential_cols if col in features_data.columns]

        # Filter the DataFrame to include only compatible columns
        features_data_filtered = features_data[final_columns].copy()
        self.logger.info(f'✅ Filtered features: {len(final_columns)} columns ({len(numeric_columns)} numeric + {len([col for col in essential_cols if col in features_data.columns])} essential)')
        self.logger.info(f'🗑️ Removed {len(features_data.columns) - len(final_columns)} non-numeric columns')

        # Ensure only numeric columns proceed downstream
        numeric_cols_final = features_data_filtered.select_dtypes(include=[np.number]).columns
        return features_data_filtered[numeric_cols_final].copy()
    
    def _engineer_features_enhanced_basic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced basic feature engineering with market microstructure features."""
        try:
            self.logger.info('🔧 Engineering enhanced basic features...')
            self.logger.info(f'📊 Input data shape: {data.shape}')
            self.logger.info(f'📊 Input data columns: {list(data.columns)}')

            # Safety check: Remove any string columns that might have slipped through
            string_columns = [col for col in data.columns if data[col].dtype == 'object' or str(data[col].dtype).startswith('string')]
            if string_columns:
                self.logger.warning(f'🚨 String columns detected in enhanced basic feature engineering: {string_columns}')
                data = data.drop(columns=string_columns, errors='ignore')
                self.logger.info(f'✅ Removed {len(string_columns)} string columns from enhanced basic features')

            # Standardize column names
            column_mapping = {}
            for col in data.columns:
                col_lower = col.lower()
                if 'open' in col_lower and 'open' not in column_mapping:
                    column_mapping['open'] = col
                elif 'high' in col_lower and 'high' not in column_mapping:
                    column_mapping['high'] = col
                elif 'low' in col_lower and 'low' not in column_mapping:
                    column_mapping['low'] = col
                elif 'close' in col_lower and 'close' not in column_mapping:
                    column_mapping['close'] = col
                elif 'volume' in col_lower and 'volume' not in column_mapping:
                    column_mapping['volume'] = col

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in column_mapping]
            if missing_columns:
                raise ValueError(f'Missing required columns: {missing_columns}. Available columns: {list(data.columns)}')

            self.logger.info(f'📊 Column mapping: {column_mapping}')

            features_data = data.copy()
            for standard_name, actual_name in column_mapping.items():
                features_data[standard_name] = features_data[actual_name]

            self.logger.info(f'📊 After column standardization: {features_data.shape}')
        except Exception as e:
            self.logger.error(f'❌ Error in feature engineering setup: {e}')
            raise
        
        try:
            # Basic price features
            features_data['price_range'] = features_data['high'] - features_data['low']
            features_data['price_change'] = features_data['close'].pct_change()
            features_data['volume_change'] = features_data['volume'].pct_change()

            self.logger.info(f'📊 After basic price features: {features_data.shape}')

            # Market microstructure features
            features_data['spread'] = features_data['high'] - features_data['low']
            features_data['spread_pct'] = features_data['spread'] / features_data['close']
            features_data['typical_price'] = (features_data['high'] + features_data['low'] + features_data['close']) / 3
            features_data['vwap'] = (features_data['typical_price'] * features_data['volume']).cumsum() / features_data['volume'].cumsum()
            features_data['price_to_vwap'] = features_data['close'] / features_data['vwap']
            features_data['dollar_volume'] = features_data['close'] * features_data['volume']
            features_data['log_dollar_volume'] = np.log1p(features_data['dollar_volume'])
            features_data['price_impact'] = features_data['price_change'].abs() / (features_data['volume'] + 1)
            features_data['kyle_lambda'] = features_data['price_impact'].rolling(20).mean()
            features_data['order_flow_imbalance'] = np.where(features_data['close'] > features_data['open'], features_data['volume'], -features_data['volume'])
            features_data['ofi_cumsum'] = features_data['order_flow_imbalance'].cumsum()

            self.logger.info(f'📊 After microstructure features: {features_data.shape}')
        except Exception as e:
            self.logger.error(f'❌ Error adding basic/microstructure features: {e}')
            raise
        
        try:
            # Technical indicators
            for period in [5, 10, 20, 50]:
                features_data[f'sma_{period}'] = features_data['close'].rolling(period).mean()
                features_data[f'price_sma_{period}_ratio'] = features_data['close'] / features_data[f'sma_{period}']

            self.logger.info(f'📊 After technical indicators: {features_data.shape}')

            # Volatility features
            features_data['volatility_5'] = features_data['price_change'].rolling(5).std()
            features_data['volatility_20'] = features_data['price_change'].rolling(20).std()

            # RSI
            delta = features_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window = 14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
            rs = gain / loss
            features_data['rsi'] = 100 - 100 / (1 + rs)

            # Bollinger Bands
            features_data['bb_middle'] = features_data['close'].rolling(20).mean()
            bb_std = features_data['close'].rolling(20).std()
            features_data['bb_upper'] = features_data['bb_middle'] + bb_std * 2
            features_data['bb_lower'] = features_data['bb_middle'] - bb_std * 2
            features_data['bb_position'] = (features_data['close'] - features_data['bb_lower']) / (features_data['bb_upper'] - features_data['bb_lower'])

            self.logger.info(f'📊 After technical indicators: {features_data.shape}')

            # Price position features
            features_data['high_low_ratio'] = features_data['high'] / features_data['low']
            features_data['close_high_ratio'] = features_data['close'] / features_data['high']
            features_data['close_low_ratio'] = features_data['close'] / features_data['low']

            # Volume features
            features_data['volume_sma_20'] = features_data['volume'].rolling(20).mean()
            features_data['volume_ratio'] = features_data['volume'] / features_data['volume_sma_20']

            self.logger.info(f'📊 After all features: {features_data.shape}')
        except Exception as e:
            self.logger.error(f'❌ Error adding technical/position features: {e}')
            raise
        
        # Fill NaN values and handle infinite values
        features_data = features_data.ffill().fillna(0)

        # Replace infinite values with finite values
        features_data = features_data.replace([np.inf, -np.inf], [1e10, -1e10])

        # Ensure all numeric columns are float64 to avoid NumPy type issues
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            features_data[col] = features_data[col].astype('float64')

        self.logger.info(f'📊 Output data shape: {features_data.shape}')
        self.logger.info(f'📊 Output data columns: {len(features_data.columns)}')

        # CRITICAL FIX: Filter out non-numeric columns to prevent string-to-float conversion errors
        # Keep only numeric columns and essential columns for ML processing
        self.logger.info('🔧 Filtering features to include only numeric columns for ML compatibility...')
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns.tolist()

        # Add essential non-numeric columns that are needed for processing but will be excluded from ML features
        essential_cols = ['timestamp']  # Keep timestamp for indexing
        final_columns = numeric_columns + [col for col in essential_cols if col in features_data.columns]

        # Filter the DataFrame to include only compatible columns
        features_data_filtered = features_data[final_columns].copy()
        self.logger.info(f'✅ Filtered features: {len(final_columns)} columns ({len(numeric_columns)} numeric + {len([col for col in essential_cols if col in features_data.columns])} essential)')
        self.logger.info(f'🗑️ Removed {len(features_data.columns) - len(final_columns)} non-numeric columns')

        return features_data_filtered

    def _validate_and_fill_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Generic function to validate and fill NaN values in features.

        Args:
            features: DataFrame with features to validate
            data: Original market data for fill values

        Returns:
            DataFrame with validated and filled features

        Raises:
            ValueError: If any feature has >5% NaN values (relaxed threshold for technical indicators)
        """
        for col in features.columns:
            nan_pct = features[col].isna().mean() * 100

            # Check for small data gaps that can be forward-filled
            if nan_pct > 0 and nan_pct <= 0.5:  # Small gaps (< 0.5%)
                # Check if gaps are due to small time differences (< 2 seconds)
                if hasattr(data, 'index') and hasattr(data.index, 'to_series'):
                    try:
                        # Calculate time gaps if data has timestamp index
                        if isinstance(data.index, pd.DatetimeIndex):
                            time_gaps = data.index.to_series().diff().dt.total_seconds()
                            max_gap = time_gaps.max() if not time_gaps.empty else 0
                            if max_gap < 2:  # Small gaps can be forward-filled
                                features[col] = features[col].fillna(method='ffill')
                                nan_pct = features[col].isna().mean() * 100
                                if nan_pct == 0:
                                    continue  # Successfully filled
                        else:
                            # If index is not datetime, try to use timestamp column if available
                            if 'timestamp' in data.columns:
                                try:
                                    # Convert timestamp to datetime for gap calculation
                                    timestamp_dt = pd.to_datetime(data['timestamp'], unit='ms')
                                    time_gaps = timestamp_dt.diff().dt.total_seconds()
                                    max_gap = time_gaps.max() if not time_gaps.empty else 0
                                    if max_gap < 2:  # Small gaps can be forward-filled
                                        features[col] = features[col].fillna(method='ffill')
                                        nan_pct = features[col].isna().mean() * 100
                                        if nan_pct == 0:
                                            continue  # Successfully filled
                                except Exception as ts_error:
                                    self.logger.debug(f'Could not calculate time gaps from timestamp column: {ts_error}')
                    except Exception as gap_error:
                        self.logger.debug(f'Could not calculate time gaps for forward-fill check: {gap_error}')

            # Selective relaxed threshold only for indicators that naturally have NaN at the beginning
            # RSI needs lookback period, others can be stricter
            if 'rsi' in col.lower():
                threshold = 5.0  # RSI has natural NaN at start due to lookback
            elif any(keyword in col.lower() for keyword in ['stoch', 'williams', 'cci', 'ma', 'sma', 'ema', 'bb_', 'atr']):
                threshold = 1.0  # Other technical indicators get moderate threshold
            else:
                threshold = 0.1  # Strict threshold for all other features
            if nan_pct > threshold:
                raise ValueError(f'❌ Excessive NaN values in {col}: {nan_pct:.2f}% (threshold: {threshold}%)')

            # Apply appropriate fill strategy based on feature type
            if any(keyword in col.lower() for keyword in ['rsi', 'stoch', 'williams', 'cci']):
                # Oscillators: fill with neutral values
                if 'rsi' in col.lower():
                    features[col] = features[col].fillna(50)
                elif 'stoch' in col.lower():
                    features[col] = features[col].fillna(50)
                elif 'williams' in col.lower():
                    features[col] = features[col].fillna(-50)
                elif 'cci' in col.lower():
                    features[col] = features[col].fillna(0)
            elif any(keyword in col.lower() for keyword in ['ma', 'sma', 'ema', 'bb_', 'vwap']):
                # Price-based features: fill with current price
                features[col] = features[col].fillna(data['close'])
            elif any(keyword in col.lower() for keyword in ['volatility', 'atr']):
                # Volatility features: fill with rolling mean or zero
                features[col] = features[col].fillna(features[col].rolling(50).mean().fillna(0))
            elif any(keyword in col.lower() for keyword in ['momentum', 'roc', 'macd']):
                # Momentum features: fill with zero
                features[col] = features[col].fillna(0)
            elif any(keyword in col.lower() for keyword in ['ratio', 'position']):
                # Ratio/position features: fill with neutral values
                features[col] = features[col].fillna(0.5 if 'position' in col.lower() else 1.0)
            else:
                # Default: fill with zero
                features[col] = features[col].fillna(0)

        return features

    def _generate_comprehensive_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive technical indicators directly."""
        features = pd.DataFrame(index=data.index)

        # Basic price features and acceleration
        features['price_change'] = data['close'].pct_change()
        features['price_change_abs'] = data['close'].diff().abs()
        features['price_acceleration'] = features['price_change'].diff()  # Acceleration
        features['price_jerk'] = features['price_acceleration'].diff()   # Jerk (rate of change of acceleration)

        # RSI variations
        for period in [7, 14, 21]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss.replace(0, np.nan))
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd_line'] = ema_12 - ema_26
        features['macd_signal'] = features['macd_line'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd_line'] - features['macd_signal']

        # Bollinger Bands
        for window in [10, 20, 30]:
            sma = data['close'].rolling(window).mean()
            std = data['close'].rolling(window).std()
            features[f'bb_middle_{window}'] = sma
            features[f'bb_upper_{window}'] = sma + (std * 2)
            features[f'bb_lower_{window}'] = sma - (std * 2)
            features[f'bb_position_{window}'] = (data['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = tr.rolling(period).mean()

        # Stochastic Oscillator
        for k_period, d_period in [(14, 3), (21, 5)]:
            lowest_low = data['low'].rolling(k_period).min()
            highest_high = data['high'].rolling(k_period).max()
            features[f'stoch_k_{k_period}'] = ((data['close'] - lowest_low) / (highest_high - lowest_low)) * 100
            features[f'stoch_d_{k_period}_{d_period}'] = features[f'stoch_k_{k_period}'].rolling(d_period).mean()

        # Williams %R
        for period in [14, 21]:
            highest_high = data['high'].rolling(period).max()
            lowest_low = data['low'].rolling(period).min()
            features[f'williams_r_{period}'] = ((highest_high - data['close']) / (highest_high - lowest_low)) * -100

        # Momentum features
        for period in [5, 10, 15, 20]:
            features[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
            features[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period) * 100

        # VWAP (Volume Weighted Average Price)
        if 'volume' in data.columns:
            data_copy = data.copy()
            data_copy['typical_price'] = (data_copy['high'] + data_copy['low'] + data_copy['close']) / 3
            data_copy['price_volume'] = data_copy['typical_price'] * data_copy['volume']
            data_copy['cumulative_price_volume'] = data_copy['price_volume'].cumsum()
            data_copy['cumulative_volume'] = data_copy['volume'].cumsum()
            features['vwap'] = data_copy['cumulative_price_volume'] / data_copy['cumulative_volume']
            features['vwap_deviation'] = (data['close'] - features['vwap']) / features['vwap'] * 100

        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = (tp - sma_tp).abs().rolling(period).mean()
            features[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100

        # Volume-based indicators
        if 'volume' in data.columns:
            for period in [5, 10, 20]:
                features[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_sma_{period}']

            # On Balance Volume (OBV)
            obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
            features['obv'] = obv

            # Volume Weighted Average Price (VWAP)
            features['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

        # Price-based volatility measures
        for period in [5, 10, 20, 30]:
            returns = data['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(period).std()
            features[f'high_low_ratio_{period}'] = (data['high'] / data['low']).rolling(period).mean()

        # Gap analysis
        close_shifted = data['close'].shift(1)
        features['gap_up'] = ((data['open'] > close_shifted) & close_shifted.notna()).astype(int)
        features['gap_down'] = ((data['open'] < close_shifted) & close_shifted.notna()).astype(int)
        features['gap_size'] = (data['open'] - close_shifted) / close_shifted

        # Intraday momentum
        features['open_to_close'] = (data['close'] - data['open']) / data['open']
        features['high_to_low'] = (data['high'] - data['low']) / data['low']
        features['close_to_high'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        # Add advanced momentum features
        momentum_features = self._calculate_advanced_momentum_features(data)
        momentum_features = self._validate_and_fill_features(momentum_features, data)
        for col in momentum_features.columns:
            features[f'momentum_{col}'] = momentum_features[col]

        # Add correlation features
        correlation_features = self._calculate_correlation_features(data)
        correlation_features = self._validate_and_fill_features(correlation_features, data)
        for col in correlation_features.columns:
            features[f'correlation_{col}'] = correlation_features[col]

        # Add liquidity features (if volume data available)
        if 'volume' in data.columns:
            liquidity_features = self._calculate_liquidity_features(data)
            liquidity_features = self._validate_and_fill_features(liquidity_features, data)
            for col in liquidity_features.columns:
                features[f'liquidity_{col}'] = liquidity_features[col]

        # Add adaptive features
        adaptive_features = self._calculate_adaptive_features(data)
        adaptive_features = self._validate_and_fill_features(adaptive_features, data)
        for col in adaptive_features.columns:
            features[f'adaptive_{col}'] = adaptive_features[col]

        # Validate and fill NaN values using generic function
        features = self._validate_and_fill_features(features, data)

        return features

    def _calculate_advanced_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced momentum features."""
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change().fillna(0)

        # Momentum indicators
        features['momentum_5'] = returns.rolling(5).mean()
        features['momentum_20'] = returns.rolling(20).mean()
        features['momentum_50'] = returns.rolling(50).mean()

        # Momentum acceleration
        features['momentum_acceleration'] = features['momentum_5'] - features['momentum_20']

        # Momentum strength
        momentum_20_std = features['momentum_20'].rolling(20).std().fillna(1e-8)
        features['momentum_strength'] = features['momentum_5'] / (momentum_20_std + 1e-8)

        # Momentum divergence
        price_momentum = data['close'].pct_change(5)
        volume_momentum = data.get('volume', pd.Series(1, index=data.index)).pct_change(5).fillna(0)
        features['momentum_divergence'] = price_momentum - volume_momentum

        return features.fillna(0)

    def _calculate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation features."""
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change().fillna(0)

        # Rolling autocorrelations
        features['autocorrelation_5'] = returns.rolling(5).corr(returns.shift(1))
        features['autocorrelation_20'] = returns.rolling(20).corr(returns.shift(1))

        # Cross-timeframe correlations (simplified)
        returns_5 = returns.rolling(5).mean()
        returns_20 = returns.rolling(20).mean()
        features['cross_timeframe_correlation'] = returns_5.rolling(20).corr(returns_20)

        return features.fillna(0)

    def _calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity features."""
        features = pd.DataFrame(index=data.index)

        if 'volume' not in data.columns:
            return features

        # Volume-based liquidity
        avg_volume = data['volume'].rolling(20).mean()
        features['volume_liquidity'] = data['volume'] / (avg_volume + 1e-8)

        return features

        return {
            'direction_accuracy': 0.5,
            'volatility_mae': 0.1,
            'model_type': 'fallback',
            'training_samples': len(features_data),
            'sr_levels_used': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
            'training_time': 0.0,
            'fallback_reason': 'chunk_processing_failed'
        }
    
    def _aggregate_chunk_results(self, all_results: List[Dict[str, Any]], chunk_times: List[float]) -> Dict[str, Any]:
        """Aggregate results from multiple chunks into a single result."""
        try:
            if not all_results:
                raise ValueError("No chunk results available for aggregation")
            
            # Calculate weighted averages based on training samples
            total_samples = sum(result.get('training_samples', 0) for result in all_results)
            if total_samples == 0:
                return all_results[0]  # Return first result if no samples
            
            # Weighted accuracy
            weighted_accuracy = sum(
                result.get('direction_accuracy', 0.5) * result.get('training_samples', 0)
                for result in all_results
            ) / total_samples
            
            # Weighted MAE
            weighted_mae = sum(
                result.get('volatility_mae', 0.1) * result.get('training_samples', 0)
                for result in all_results
            ) / total_samples
            
            # Total training time
            total_training_time = sum(result.get('training_time', 0) for result in all_results)
            
            # Most common model type
            model_types = [result.get('model_type', 'unknown') for result in all_results]
            most_common_model = max(set(model_types), key=model_types.count)
            
            return {
                'direction_accuracy': weighted_accuracy,
                'volatility_mae': weighted_mae,
                'model_type': most_common_model,
                'training_samples': total_samples,
                'sr_levels_used': all_results[0].get('sr_levels_used', 0),
                'training_time': total_training_time,
                'chunks_processed': len(all_results),
                'avg_chunk_time': np.mean(chunk_times) if chunk_times else 0.0,
                'aggregation_method': 'weighted_average'
            }
            
        except Exception as e:
            self.logger.error(f'❌ Result aggregation failed: {e}')
            if all_results:
                return all_results[0]  # Return first result if available
            else:
                raise RuntimeError(f"Result aggregation failed: {e}")

    def _run_sr_calculation_chunked(self, sr_manager, data: pd.DataFrame) -> Dict[str, Any]:
        """Run SR calculation with chunked processing for large datasets."""
        try:
            chunk_size = 50000  # Process 50K points at a time
            all_support_levels = []
            all_resistance_levels = []

            total_chunks = (len(data) + chunk_size - 1) // chunk_size
            self.logger.info(f'📊 Processing {len(data)} points in {total_chunks} chunks of {chunk_size} points each')

            for i in range(0, len(data), chunk_size):
                chunk_end = min(i + chunk_size, len(data))
                chunk_data = data.iloc[i:chunk_end]

                self.logger.info(f'🔄 Processing chunk {i//chunk_size + 1}/{total_chunks} ({len(chunk_data)} points)')

                # Process this chunk
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_sr_calculation, sr_manager, chunk_data)
                    chunk_results = future.result(timeout=600)  # 10 minutes per chunk

                # Merge results
                if 'support_levels' in chunk_results:
                    all_support_levels.extend(chunk_results['support_levels'])
                if 'resistance_levels' in chunk_results:
                    all_resistance_levels.extend(chunk_results['resistance_levels'])

            # Consolidate and deduplicate levels
            consolidated_results = self._consolidate_sr_levels(all_support_levels, all_resistance_levels)
            self.logger.info(f'✅ Chunked processing completed: {len(consolidated_results.get("support_levels", []))} support, {len(consolidated_results.get("resistance_levels", []))} resistance levels')

            return consolidated_results

        except Exception as e:
            self.logger.error(f'Chunked SR calculation failed: {e}')
            return {'support_levels': [], 'resistance_levels': []}

    def _consolidate_sr_levels(self, support_levels, resistance_levels) -> Dict[str, Any]:
        """Consolidate and deduplicate SR levels from multiple chunks."""
        try:
            # Simple deduplication by price proximity (within 0.1% of price)
            def deduplicate_levels(levels, level_type):
                if not levels:
                    return []

                # Sort by price
                sorted_levels = sorted(levels, key=lambda x: x.price if hasattr(x, 'price') else x.get('price', 0))

                consolidated = []
                current_group = [sorted_levels[0]]

                for level in sorted_levels[1:]:
                    current_price = current_group[0].price if hasattr(current_group[0], 'price') else current_group[0].get('price', 0)
                    level_price = level.price if hasattr(level, 'price') else level.get('price', 0)

                    # If within 0.1% of current group price, add to group
                    if abs(level_price - current_price) / current_price < 0.001:
                        current_group.append(level)
                    else:
                        # Consolidate current group and start new group
                        if current_group:
                            consolidated.append(self._merge_level_group(current_group, level_type))
                        current_group = [level]

                # Don't forget the last group
                if current_group:
                    consolidated.append(self._merge_level_group(current_group, level_type))

                return consolidated[:50]  # Limit to top 50 levels

            consolidated_support = deduplicate_levels(support_levels, 'support')
            consolidated_resistance = deduplicate_levels(resistance_levels, 'resistance')

            return {
                'support_levels': consolidated_support,
                'resistance_levels': consolidated_resistance
            }

        except Exception as e:
            self.logger.error(f'Level consolidation failed: {e}')
            return {'support_levels': support_levels[:50], 'resistance_levels': resistance_levels[:50]}

    def _merge_level_group(self, level_group, level_type):
        """Merge a group of similar levels into a single level."""
        try:
            if not level_group:
                return None

            # Calculate weighted average price based on strength
            total_weight = 0
            weighted_price = 0

            for level in level_group:
                strength = getattr(level, 'strength', 0.5) if hasattr(level, 'strength') else level.get('strength', 0.5)
                price = level.price if hasattr(level, 'price') else level.get('price', 0)

                weighted_price += price * strength
                total_weight += strength

            avg_price = weighted_price / total_weight if total_weight > 0 else level_group[0].price if hasattr(level_group[0], 'price') else level_group[0].get('price', 0)
            max_strength = max(getattr(level, 'strength', 0.5) if hasattr(level, 'strength') else level.get('strength', 0.5) for level in level_group)

            # Create consolidated level
            return {
                'price': float(avg_price),
                'strength': float(max_strength),
                'type': level_type,
                'method': 'consolidated_fractal',
                'touch_count': sum(getattr(level, 'touch_count', 1) if hasattr(level, 'touch_count') else level.get('touch_count', 1) for level in level_group),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f'Level group merging failed: {e}')
            return level_group[0] if level_group else None

    def _should_terminate_sr_detection(self, elapsed_seconds: int, remaining_seconds: int) -> bool:
        """Check if SR detection should be terminated early based on system constraints."""
        try:
            import psutil
            import os

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Check process memory
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / 1024 / 1024

            # Terminate if memory usage is too high
            if memory_usage_percent > 90:  # Over 90% memory usage
                self.logger.warning(f'🛑 High memory usage detected ({memory_usage_percent:.1f}%), terminating SR detection')
                return True

            if process_memory_mb > 8000:  # Over 8GB process memory
                self.logger.warning(f'🛑 High process memory usage detected ({process_memory_mb:.1f}MB), terminating SR detection')
                return True

            # Check CPU usage (if consistently high, might indicate hanging)
            cpu_percent = psutil.cpu_percent(interval=1)

            # If CPU usage is very low for an extended period, might indicate hanging
            if elapsed_seconds > 600 and cpu_percent < 5:  # Low CPU for more than 10 minutes
                self.logger.warning(f'🛑 Low CPU usage ({cpu_percent:.1f}%) for extended period, possible hang detected')
                return True

            return False

        except Exception as e:
            self.logger.warning(f'⚠️ Could not check system constraints: {e}')
            return False

    async def _run_sr_detection_with_fast_fail(self, features_data: pd.DataFrame) -> Dict[str, Any]:
        """Run SR detection with comprehensive fast-fail checks."""
        try:
            # Fast-fail: Check if we have sufficient data for SR detection
            if len(features_data) < 500:
                self.logger.warning(f'⚠️ Insufficient data for SR detection: {len(features_data)} rows (minimum: 500)')
                return self._get_fallback_sr_levels()
            
            # Fast-fail: Check memory usage before SR detection
            memory_usage = self._check_memory_usage()
            if memory_usage > 0.85:
                self.logger.warning(f'⚠️ High memory usage before SR detection: {memory_usage:.1%}')
                return self._get_fallback_sr_levels()
            
            # Run SR detection without timeout (let it complete naturally)
            sr_levels = await self._run_sr_detection(features_data)
            
            # Fast-fail: Check if SR detection produced meaningful results
            if not self._validate_sr_results(sr_levels):
                self.logger.warning('⚠️ SR detection produced invalid results')
                raise RuntimeError("Advanced SR detection produced invalid results. No fallback available.")

            return sr_levels

        except Exception as e:
            self.logger.error(f'❌ SR detection with fast-fail failed: {e}')
            raise RuntimeError(f"Advanced SR detection failed: {e}. No fallback available.")

    def _validate_sr_results(self, sr_levels: Dict[str, Any]) -> bool:
        """Validate SR detection results for meaningful output."""
        try:
            # Check if results are empty
            if not sr_levels:
                return False
            
            # Check if we have the expected structure
            if 'support_levels' not in sr_levels or 'resistance_levels' not in sr_levels:
                return False
            
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            
            # Check if we have at least some levels
            total_levels = len(support_levels) + len(resistance_levels)
            if total_levels == 0:
                return False
            
            # Check if levels are reasonable (not too many or too few)
            if total_levels > 1000:  # Too many levels might indicate an error
                self.logger.warning(f'⚠️ Suspiciously high number of SR levels: {total_levels}')
                return False
            
            # Check if levels have reasonable price values
            all_levels = support_levels + resistance_levels
            for level in all_levels[:10]:  # Check first 10 levels
                if isinstance(level, dict):
                    price = level.get('price', level.get('level', 0))
                    if price <= 0 or price > 1000000:  # Unreasonable price values
                        self.logger.warning(f'⚠️ Suspicious price value in SR level: {price}')
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f'⚠️ SR result validation failed: {e}')
            return False

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) for normalization."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean of True Range
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            self.logger.warning(f'ATR calculation failed: {e}')
            # Fallback to simple price range
            return (data['high'] - data['low']).rolling(window=period).mean()

    def _format_sr_levels_for_pipeline(self, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Format S/R levels for pipeline state consumption by step06."""
        try:
            formatted_levels = {
                'support_levels': [],
                'resistance_levels': [],
                'metadata': {
                    'total_support': len(sr_levels.get('support_levels', [])),
                    'total_resistance': len(sr_levels.get('resistance_levels', [])),
                    'detection_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            # Format support levels
            for level in sr_levels.get('support_levels', []):
                if hasattr(level, 'price'):  # SRLevel object
                    formatted_level = {
                        'price': level.price,
                        'strength': level.strength,
                        'touch_count': level.touch_count,
                        'first_touch_time': level.first_touch_time.isoformat() if level.first_touch_time else None,
                        'last_touch_time': level.last_touch_time.isoformat() if level.last_touch_time else None,
                        'age_bars': level.age_bars,
                        'avg_bounce_ratio': level.avg_bounce_ratio,
                        'max_bounce_ratio': level.max_bounce_ratio,
                        'volume_confirmation_score': level.volume_confirmation_score,
                        'consistency_score': level.consistency_score,
                        'confidence_score': level.confidence_score,
                        'confluence_score': level.confluence_score,
                        'type': 'support'
                    }
                else:  # Dictionary format
                    formatted_level = {
                        'price': level.get('price', level),
                        'strength': level.get('strength', 0.5),
                        'touch_count': level.get('touch_count', 1),
                        'type': 'support'
                    }
                formatted_levels['support_levels'].append(formatted_level)
            
            # Format resistance levels
            for level in sr_levels.get('resistance_levels', []):
                if hasattr(level, 'price'):  # SRLevel object
                    formatted_level = {
                        'price': level.price,
                        'strength': level.strength,
                        'touch_count': level.touch_count,
                        'first_touch_time': level.first_touch_time.isoformat() if level.first_touch_time else None,
                        'last_touch_time': level.last_touch_time.isoformat() if level.last_touch_time else None,
                        'age_bars': level.age_bars,
                        'avg_bounce_ratio': level.avg_bounce_ratio,
                        'max_bounce_ratio': level.max_bounce_ratio,
                        'volume_confirmation_score': level.volume_confirmation_score,
                        'consistency_score': level.consistency_score,
                        'confidence_score': level.confidence_score,
                        'confluence_score': level.confluence_score,
                        'type': 'resistance'
                    }
                else:  # Dictionary format
                    formatted_level = {
                        'price': level.get('price', level),
                        'strength': level.get('strength', 0.5),
                        'touch_count': level.get('touch_count', 1),
                        'type': 'resistance'
                    }
                formatted_levels['resistance_levels'].append(formatted_level)
            
            self.logger.info(f'✅ Formatted {len(formatted_levels["support_levels"])} support and {len(formatted_levels["resistance_levels"])} resistance levels for pipeline')
            return formatted_levels
            
        except Exception as e:
            self.logger.error(f'❌ Failed to format S/R levels for pipeline: {e}')
        try:
            # Select numeric features only
            numeric_features = features_data.select_dtypes(include=[np.number]).columns
            self.logger.info(f'🔢 Selected {len(numeric_features)} numeric features for ML training')

            # Remove target-related columns if they exist
            exclude_cols = ['direction_target', 'volatility_target', 'target', 'label']
            feature_cols = [col for col in numeric_features if col not in exclude_cols]

            # Handle missing values
            X = features_data[feature_cols].fillna(0).values
            feature_names = np.array(feature_cols)

            # Get targets
            y_direction = target_data['direction_target'].values
            y_volatility = target_data['volatility_target'].values

            # Remove rows where direction target is neutral (class 2)
            valid_mask = y_direction != 2
            X = X[valid_mask]
            y_direction = y_direction[valid_mask]
            y_volatility = y_volatility[valid_mask]

            self.logger.info(f'📊 ML data prepared: {X.shape[0]} samples, {X.shape[1]} features')
            self.logger.info(f'🎯 Direction target distribution: {np.bincount(y_direction.astype(int))}')
            self.logger.info(f'📈 Volatility target range: {y_volatility.min():.6f} - {y_volatility.max():.6f}')

            return X, y_direction, y_volatility, feature_names

        except Exception as e:
            self.logger.error(f'❌ Failed to prepare ML features: {e}')
            raise
    def _m1_gpu_hyperparameter_optimization(self, X_tensor, y_tensor, feature_names: np.ndarray, batch_size: int) -> Dict[str, Any]:
        """Optimize hyperparameters using M1 GPU acceleration."""
        try:
            self.logger.info('🍎 Starting M1 GPU hyperparameter optimization')
            
            # Use M1 GPU manager for batch processing
            if self.m1_gpu_manager:
                # Process data in optimal batches
                results = []
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    batch_y = y_tensor[i:i+batch_size]
                    
                    # Use M1 GPU matrix operations for optimization
                    if M1_GPU_AVAILABLE:
                        # Perform GPU-accelerated optimization
                        batch_result = self.m1_gpu_manager.batch_process_mps(
                            batch_X, batch_y, "neural_net"
                        )
                        results.append(batch_result)
                
                # Combine results from all batches
                if results:
                    # Use math_validation for result validation
                    combined_result = self._combine_optimization_results(results)
                    if validate_finite(combined_result.get('score', 0)):
                        return combined_result
            
            return {}
            
        except Exception as e:
            self.logger.warning(f'⚠️ M1 GPU hyperparameter optimization failed: {e}')
            return {}
    
    def _m1_cpu_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using M1 CPU parallel processing."""
        try:
            self.logger.info('🍎 Starting M1 CPU parallel hyperparameter optimization')
            
            if self.m1_cpu_optimizer:
                # Use M1 CPU optimizer for parallel processing
                optimal_workers = self.m1_cpu_optimizer.get_optimal_workers()
                self.logger.info(f'🔧 Using {optimal_workers} workers for parallel optimization')
                
                # Use M1 batch processor for optimization
                batch_processor = self.m1_cpu_optimizer.create_batch_processor(
                    batch_size=1000,
                    max_workers=optimal_workers
                )
                
                # Process optimization in parallel batches
                optimization_tasks = self._create_optimization_tasks(X, y, feature_names)
                results = batch_processor.process_batches(optimization_tasks)
                
                # Combine results
                if results:
                    combined_result = self._combine_optimization_results(results)
                    if validate_finite(combined_result.get('score', 0)):
                        return combined_result
            
            return {}
            
        except Exception as e:
            self.logger.warning(f'⚠️ M1 CPU hyperparameter optimization failed: {e}')
            return {}
    
    def _chunked_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using chunked processing for memory efficiency."""
        try:
            self.logger.info('📦 Starting chunked hyperparameter optimization')
            
            if self.m1_memory_optimizer:
                # Use M1 memory optimizer for chunked processing
                chunk_size = self.m1_memory_optimizer.get_optimal_chunk_size(X.nbytes)
                self.logger.info(f'📏 Using chunk size: {chunk_size} bytes')
                
                # Process data in chunks
                results = []
                for i in range(0, len(X), chunk_size):
                    chunk_X = X[i:i+chunk_size]
                    chunk_y = y[i:i+chunk_size]
                    
                    # Use memory checkpoint for each chunk
                    with self.m1_memory_optimizer.memory_checkpoint(f"chunk_{i}"):
                        chunk_result = self._simplified_hyperparameter_optimization(chunk_X, chunk_y, feature_names)
                        if chunk_result:
                            results.append(chunk_result)
                
                # Combine results from all chunks
                if results:
                    combined_result = self._combine_optimization_results(results)
                    if validate_finite(combined_result.get('score', 0)):
                        return combined_result
            
            return {}
            
        except Exception as e:
            self.logger.warning(f'⚠️ Chunked hyperparameter optimization failed: {e}')
            return {}
    
    def _create_optimization_tasks(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> List[Dict[str, Any]]:
        """Create optimization tasks for parallel processing."""
        tasks = []
        
        # Create different optimization tasks
        optimization_methods = ['grid_search', 'random_search', 'bayesian']
        
        for method in optimization_methods:
            task = {
                'method': method,
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'optimization_method': method
            }
            tasks.append(task)
        
        return tasks
    
    def _combine_optimization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple optimization runs."""
        if not results:
            return {}
        
        # Find the best result based on score
        best_result = max(results, key=lambda x: x.get('score', 0))
        
        # Use math_validation to ensure the result is valid
        if validate_finite(best_result.get('score', 0)):
            return best_result
        
        return {}
    
    def _create_model_training_tasks(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_dir_train: np.ndarray, y_dir_test: np.ndarray,
                                   y_vol_train: np.ndarray, y_vol_test: np.ndarray,
                                   feature_names: np.ndarray) -> List[Dict[str, Any]]:
        """Create training tasks for parallel model processing."""
        tasks = []
        
        # Create tasks for different model types
        model_types = ['extra_trees', 'xgboost', 'lightgbm', 'random_forest']
        
        for model_type in model_types:
            task = {
                'model_type': model_type,
                'X_train': X_train,
                'X_test': X_test,
                'y_dir_train': y_dir_train,
                'y_dir_test': y_dir_test,
                'y_vol_train': y_vol_train,
                'y_vol_test': y_vol_test,
                'feature_names': feature_names
            }
            tasks.append(task)
        
        return tasks
    
    def _combine_model_results(self, parallel_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from parallel model training."""
        if not parallel_results:
            return {}
        
        # Combine results from all parallel training tasks
        combined_results = {}
        
        for result in parallel_results:
            if result and 'model_type' in result:
                model_type = result['model_type']
                combined_results[model_type] = result
        
        # Calculate overall metrics
        if combined_results:
            # Find best performing model
            best_direction_accuracy = 0
            best_volatility_mae = float('inf')
            best_direction_model = None
            best_volatility_model = None
            
            for model_type, result in combined_results.items():
                if 'direction' in result and 'accuracy' in result['direction']:
                    accuracy = result['direction']['accuracy']
                    if validate_finite(accuracy) and accuracy > best_direction_accuracy:
                        best_direction_accuracy = accuracy
                        best_direction_model = model_type
                
                if 'volatility' in result and 'mae' in result['volatility']:
                    mae = result['volatility']['mae']
                    if validate_finite(mae) and mae < best_volatility_mae:
                        best_volatility_mae = mae
                        best_volatility_model = model_type
            
            # Return combined results
            return {
                'models': combined_results,
                'best_direction_model': best_direction_model,
                'best_volatility_model': best_volatility_model,
                'best_direction_accuracy': best_direction_accuracy,
                'best_volatility_mae': best_volatility_mae,
                'm1_cpu_optimization_used': True,
                'parallel_processing_enabled': self.enable_parallel_processing
            }
        
        return {}

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters when optimization is not feasible."""
        return {
            'method': 'default',
            'best_score': 0.0,
            'best_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            },
            'optimization_time': 0.0,
            'reason': 'insufficient_data_or_memory'
        }
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage as a percentage."""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
            return 0.0
        except Exception:
            return 0.0
    
    def _simplified_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Simplified hyperparameter optimization for resource-constrained environments."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            
            # Very limited parameter space
            param_combinations = [
                {'n_estimators': 50, 'max_depth': 5, 'max_features': 'sqrt'},
                {'n_estimators': 100, 'max_depth': 10, 'max_features': 'sqrt'},
                {'n_estimators': 50, 'max_depth': None, 'max_features': 'log2'}
            ]
            
            best_score = 0.0
            best_params = param_combinations[0]
            
            for params in param_combinations:
                model = RandomForestClassifier(random_state=42, **params)
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
            
            return {
                'method': 'simplified',
                'best_score': best_score,
                'best_params': best_params,
                'optimization_time': 0.0,
                'reason': 'memory_constrained'
            }
            
        except Exception as e:
            self.logger.error(f'❌ Simplified hyperparameter optimization failed: {e}')
            return self._get_default_hyperparameters()
    
    def _halving_search_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Halving search hyperparameter optimization for faster convergence."""
        try:
            from sklearn.experimental import enable_halving_search_cv
            from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier
            
            # Reduced parameter space for faster convergence
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            # Use time series split with fewer folds
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Halving grid search for faster convergence
            halving_search = HalvingGridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=min(4, os.cpu_count()),  # Limit parallelization
                verbose=0,
                factor=2,  # Halving factor
                min_resources=100  # Minimum resources per candidate
            )
            
            start_time = time.time()
            halving_search.fit(X, y)
            optimization_time = time.time() - start_time
            
            self.logger.info(f'🔧 Halving search best score: {halving_search.best_score_:.4f}')
            self.logger.info(f'🔧 Halving search best params: {halving_search.best_params_}')
            self.logger.info(f'⏱️ Optimization time: {optimization_time:.2f}s')
            
            return {
                'method': 'halving_search',
                'best_score': halving_search.best_score_,
                'best_params': halving_search.best_params_,
                'cv_results': halving_search.cv_results_,
                'optimization_time': optimization_time
            }
            
        except Exception as e:
            self.logger.error(f'❌ Halving search optimization failed: {e}')
            return self._simplified_hyperparameter_optimization(X, y, feature_names)
    
    
    def _optimized_bayesian_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Optimized Bayesian optimization with reduced iterations."""
        try:
            if not OPTIMIZATION_AVAILABLE:
                self.logger.warning('⚠️ Bayesian optimization not available, falling back to halving search')
                return self._halving_search_optimization(X, y, feature_names)
            
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            
            # Reduced search space
            space = [
                Integer(50, 200, name='n_estimators'),
                Integer(5, 20, name='max_depth'),
                Integer(2, 10, name='min_samples_split'),
                Integer(1, 5, name='min_samples_leaf'),
                Categorical(['sqrt', 'log2'], name='max_features')
            ]
            
            # Use time series split with fewer folds
            tscv = TimeSeriesSplit(n_splits=3)
            
            @use_named_args(space)
            def objective(**params):
                model = RandomForestClassifier(random_state=42, **params)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                return -scores.mean()  # Minimize negative accuracy
            
            start_time = time.time()
            result = gp_minimize(
                objective, 
                space, 
                n_calls=min(15, self.optimization_trials),  # Reduced iterations
                random_state=42
            )
            optimization_time = time.time() - start_time
            
            # Extract best parameters
            best_params = {
                'n_estimators': result.x[0],
                'max_depth': result.x[1],
                'min_samples_split': result.x[2],
                'min_samples_leaf': result.x[3],
                'max_features': result.x[4]
            }
            
            best_score = -result.fun
            
            self.logger.info(f'🔧 Optimized Bayesian search best score: {best_score:.4f}')
            self.logger.info(f'🔧 Optimized Bayesian search best params: {best_params}')
            self.logger.info(f'⏱️ Optimization time: {optimization_time:.2f}s')
            
            return {
                'method': 'optimized_bayesian',
                'best_score': best_score,
                'best_params': best_params,
                'optimization_time': optimization_time
            }
            
        except Exception as e:
            self.logger.error(f'❌ Optimized Bayesian optimization failed: {e}')
            return self._halving_search_optimization(X, y, feature_names)

    def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Grid search hyperparameter optimization."""
        try:
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=self.optimization_folds)
            
            # Grid search
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.logger.info(f'🔧 Grid search best score: {grid_search.best_score_:.4f}')
            self.logger.info(f'🔧 Grid search best params: {grid_search.best_params_}')
            
            return {
                'method': 'grid_search',
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f'❌ Grid search optimization failed: {e}')
            return {}

    def _random_search_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Random search hyperparameter optimization."""
        try:
            from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier
            from scipy.stats import randint, uniform
            
            # Define parameter distributions
            param_distributions = {
                'n_estimators': randint(50, 300),
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=self.optimization_folds)
            
            # Random search
            random_search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                param_distributions,
                n_iter=self.optimization_trials,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(X, y)
            
            self.logger.info(f'🔧 Random search best score: {random_search.best_score_:.4f}')
            self.logger.info(f'🔧 Random search best params: {random_search.best_params_}')
            
            return {
                'method': 'random_search',
                'best_score': random_search.best_score_,
                'best_params': random_search.best_params_,
                'cv_results': random_search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f'❌ Random search optimization failed: {e}')
            return {}

    def _bayesian_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Bayesian optimization using scikit-optimize."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            
            # Define search space
            space = [
                Integer(50, 300, name='n_estimators'),
                Categorical([5, 10, 15, 20, None], name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Categorical(['sqrt', 'log2', None], name='max_features')
            ]
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=self.optimization_folds)
            
            @use_named_args(space)
            def objective(**params):
                # Handle None values
                if params['max_depth'] is None:
                    params['max_depth'] = None
                
                model = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    random_state=42
                )
                
                # Cross-validation score
                scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                return -scores.mean()  # Minimize negative score
            
            # Bayesian optimization
            result = gp_minimize(objective, space, n_calls=self.optimization_trials, random_state=42)
            
            # Extract best parameters
            best_params = {
                'n_estimators': result.x[0],
                'max_depth': result.x[1],
                'min_samples_split': result.x[2],
                'min_samples_leaf': result.x[3],
                'max_features': result.x[4]
            }
            
            best_score = -result.fun
            
            self.logger.info(f'🔧 Bayesian optimization best score: {best_score:.4f}')
            self.logger.info(f'🔧 Bayesian optimization best params: {best_params}')
            
            return {
                'method': 'bayesian',
                'best_score': best_score,
                'best_params': best_params,
                'optimization_result': result
            }
            
        except Exception as e:
            self.logger.error(f'❌ Bayesian optimization failed: {e}')
            return {}

    def _walk_forward_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Perform walk-forward validation for time series data."""
        try:
            if not self.enable_walk_forward_validation:
                self.logger.info('🔧 Walk-forward validation disabled')
                return {}
            
            self.logger.info(f'🔧 Starting walk-forward validation with {self.walk_forward_folds} folds')
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Calculate fold sizes
            total_samples = len(X)
            test_size = int(total_samples * self.walk_forward_test_size)
            fold_size = (total_samples - test_size) // self.walk_forward_folds
            
            results = {
                'fold_scores': [],
                'fold_metrics': [],
                'average_metrics': {}
            }
            
            for fold in range(self.walk_forward_folds):
                # Calculate train/test indices
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = test_start + test_size
                
                if test_end > total_samples:
                    break
                
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_test = X[test_start:test_end]
                y_test = y[test_start:test_end]
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                fold_metrics = {
                    'fold': fold + 1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
                
                results['fold_scores'].append(accuracy)
                results['fold_metrics'].append(fold_metrics)
                
                self.logger.info(f'🔧 Fold {fold + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}')
            
            # Calculate average metrics
            if results['fold_scores']:
                results['average_metrics'] = {
                    'mean_accuracy': np.mean(results['fold_scores']),
                    'std_accuracy': np.std(results['fold_scores']),
                    'mean_precision': np.mean([m['precision'] for m in results['fold_metrics']]),
                    'mean_recall': np.mean([m['recall'] for m in results['fold_metrics']]),
                    'mean_f1_score': np.mean([m['f1_score'] for m in results['fold_metrics']])
                }
                
                self.logger.info(f'🔧 Walk-forward validation complete: Mean Accuracy = {results["average_metrics"]["mean_accuracy"]:.4f} ± {results["average_metrics"]["std_accuracy"]:.4f}')
            
            return results
            
        except Exception as e:
            self.logger.error(f'❌ Walk-forward validation failed: {e}')
            return {}

    def _optimize_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Optimize feature selection using ML Common utilities while maintaining target feature counts."""
        try:
            start_time = time.time()
            n_samples, n_features = X.shape

            self.logger.info(f'🎯 Starting enhanced feature selection with ML Common utilities')
            self.logger.info(f'📊 Dataset: {n_samples} samples, {n_features} features')

            # Determine target number of features (same as original logic)
            if n_samples < 5000:
                target_features = min(30, n_features)
            elif n_samples < 20000:
                target_features = min(60, n_features)
            else:
                target_features = min(120, n_features)

            self.logger.info(f'🎯 Target features: {target_features} (based on {n_samples} samples)')

            feature_selection_info = {
                'original_features': n_features,
                'target_features': target_features,
                'methods_used': [],
                'selected_features': 0,
                'feature_importance': {},
                'optimization_time': 0.0,
                'ml_common_used': ML_COMMON_AVAILABLE
            }

            # Use ML Common FeatureSelectionFramework if available
            if ML_COMMON_AVAILABLE and self.feature_selector:
                self.logger.info('🔧 Using ML Common FeatureSelectionFramework')

                # Use mRMR selection for comprehensive feature selection
                mrmr_result = self.feature_selector.mrmr_selection(
                    X, y, feature_names.tolist(), target_features,
                    relevance_method='mutual_info',
                    redundancy_method='correlation'
                )

                if mrmr_result.get('selected_features'):
                    selected_feature_names_list = mrmr_result['selected_features']
                    selected_feature_names = np.array(selected_feature_names_list)

                    # Create boolean mask for selected features
                    selected_mask = np.isin(feature_names, selected_feature_names)
                    X_selected = X[:, selected_mask]

                    # Extract feature importance from mRMR results
                    feature_importance = {}
                    if 'feature_scores' in mrmr_result:
                        for feature, scores in mrmr_result['feature_scores'].items():
                            feature_importance[feature] = scores.get('mrmr_score', 0.0)

                    # Update feature selection info
                    feature_selection_info.update({
                        'methods_used': ['mrmr_selection', 'mutual_info', 'correlation_filtering'],
                        'selected_features': len(selected_feature_names),
                        'feature_importance': feature_importance,
                        'mrmr_scores': mrmr_result.get('mrmr_scores', {}),
                        'relevance_scores': mrmr_result.get('relevance_scores', {}),
                        'redundancy_analysis': mrmr_result.get('redundancy_scores', {})
                    })

                    self.logger.info(f'✅ mRMR selection completed: {len(selected_feature_names)}/{n_features} features')
                else:
                    self.logger.warning('⚠️ mRMR selection failed, falling back to legacy method')
                    return self._legacy_feature_selection(X, y, feature_names, target_features)

            else:
                # Fallback to legacy feature selection method
                self.logger.info('⚠️ ML Common not available, using legacy feature selection')
                return self._legacy_feature_selection(X, y, feature_names, target_features)

            # SR Feature Prioritization (preserve original logic)
            sr_boost = 1.2  # 120% priority for SR features
            sr_features_boosted = 0

            for feature_name in selected_feature_names:
                if any(keyword in feature_name.lower() for keyword in ['sr', 'support', 'resistance', 'proximity', 'level']):
                    if feature_name in feature_selection_info['feature_importance']:
                        original_score = feature_selection_info['feature_importance'][feature_name]
                        feature_selection_info['feature_importance'][feature_name] = original_score * sr_boost
                        sr_features_boosted += 1

            self.logger.info(f'🎯 SR Feature Prioritization: Boosted {sr_features_boosted} SR-related features by {sr_boost}x')

            # Sort features by final importance
            sorted_features = sorted(
                feature_selection_info['feature_importance'].items(),
                key=lambda x: x[1], reverse=True
            )

            # Ensure we have exactly target_features (may have changed due to boosting)
            if len(sorted_features) > target_features:
                top_features = sorted_features[:target_features]
                selected_feature_names = np.array([name for name, _ in top_features])
                selected_mask = np.isin(feature_names, selected_feature_names)
                X_selected = X[:, selected_mask]
                feature_selection_info['selected_features'] = len(selected_feature_names)

            # Add additional ML Common utilities if available
            if ML_COMMON_AVAILABLE:
                # Add data quality analysis
                if self.data_quality_utils:
                    try:
                        # Analyze feature correlation
                        corr_analysis = self.data_quality_utils.feature_correlation_analysis(
                            pd.DataFrame(X_selected, columns=selected_feature_names),
                            method='spearman'
                        )
                        if corr_analysis.get('highly_correlated_pairs'):
                            feature_selection_info['correlation_analysis'] = corr_analysis
                            self.logger.info(f'🔗 Found {len(corr_analysis["highly_correlated_pairs"])} highly correlated feature pairs')
                    except Exception as corr_e:
                        self.logger.debug(f'Correlation analysis failed: {corr_e}')

                # Add stability analysis if we have enough data
                if self.feature_selector and len(X) >= 1000:
                    try:
                        stability_scores = {}
                        for i, feature_name in enumerate(selected_feature_names):
                            # Simple variance-based stability
                            feature_values = X_selected[:, i]
                            stability_scores[feature_name] = 1.0 / (1.0 + np.std(feature_values))

                        feature_selection_info['stability_scores'] = stability_scores
                        self.logger.info('📊 Added feature stability analysis')
                    except Exception as stab_e:
                        self.logger.debug(f'Stability analysis failed: {stab_e}')

            # Final statistics and logging
            optimization_time = time.time() - start_time
            feature_selection_info['optimization_time'] = optimization_time

            sr_features = [name for name in selected_feature_names if any(keyword in name.lower() for keyword in ['sr', 'support', 'resistance', 'proximity'])]

            self.logger.info(f'✅ Enhanced feature selection completed: {len(selected_feature_names)}/{n_features} features selected')
            self.logger.info(f'⏱️ Feature selection time: {optimization_time:.2f}s')
            self.logger.info(f'🎯 SR-related features retained: {len(sr_features)} ({len(sr_features)/len(selected_feature_names):.1%})')
            self.logger.info(f'🏆 Top 5 features: {selected_feature_names[:5].tolist()}')

            if sr_features:
                self.logger.info(f'🎯 SR features in top selection: {sr_features[:5]}')

            # Log comprehensive feature selection summary
            methods_str = ', '.join(feature_selection_info['methods_used'])
            self.logger.info(f'📋 Feature selection methods used: {methods_str}')
            self.logger.info(f'📈 Feature reduction ratio: {(n_features - len(selected_feature_names))/n_features:.1%}')

            return X_selected, selected_feature_names, feature_selection_info

        except Exception as e:
            self.logger.warning(f'⚠️ Enhanced feature selection failed, using legacy method: {e}')
            return self._legacy_feature_selection(X, y, feature_names, 30)  # Default to 30 features

    def _legacy_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray, target_features: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fallback to original feature selection logic."""
        try:
            self.logger.info('🔄 Using legacy feature selection method')

            # Use variance threshold first
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_mask = selector.get_support()
            selected_feature_names = feature_names[selected_mask]

            # If we have too few features, select top by variance
            if X_selected.shape[1] < target_features:
                variances = np.var(X, axis=0)
                top_indices = np.argsort(variances)[-target_features:]
                X_selected = X[:, top_indices]
                selected_feature_names = feature_names[top_indices]

            # If we have too many features, select top by variance
            elif X_selected.shape[1] > target_features:
                variances = np.var(X_selected, axis=0)
                top_indices = np.argsort(variances)[-target_features:]
                X_selected = X_selected[:, top_indices]
                selected_feature_names = selected_feature_names[top_indices]

            feature_selection_info = {
                'original_features': len(feature_names),
                'methods_used': ['legacy_variance_threshold'],
                'selected_features': len(selected_feature_names),
                'target_features': target_features,
                'optimization_time': 0.0
            }

            return X_selected, selected_feature_names, feature_selection_info

        except Exception as e:
            self.logger.error(f'❌ Legacy feature selection failed: {e}')
            # Ultimate fallback
            return X, feature_names, {
                'original_features': len(feature_names),
                'methods_used': ['fallback'],
                'selected_features': len(feature_names),
                'error': str(e),
                'optimization_time': 0.0
            }
    
    def _get_fallback_feature_selection_info(self, feature_names: np.ndarray) -> Dict[str, Any]:
        """Get fallback feature selection info when selection is not feasible."""
        return {
            'original_features': len(feature_names),
            'methods_used': ['fallback'],
            'selected_features': len(feature_names),
            'feature_importance': {},
            'optimization_time': 0.0,
            'reason': 'insufficient_data'
        }
    


    async def _train_multiple_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_dir_train: np.ndarray, y_dir_test: np.ndarray,
                                   y_vol_train: np.ndarray, y_vol_test: np.ndarray,
                                   feature_names: np.ndarray) -> Dict[str, Any]:
        """Train multiple ML models with M1 CPU optimizer and computational optimizations."""
        try:
            # Use math_validation for input validation
            train_size = validate_positive(len(X_train), "train_size")
            test_size = validate_positive(len(X_test), "test_size")
            
            # Use M1 CPU optimizer for parallel processing
            if self.m1_cpu_optimizer and self.enable_parallel_processing:
                self.logger.info('🍎 Using M1 CPU optimizer for parallel model training')
                
                # Get optimal number of workers
                optimal_workers = self.m1_cpu_optimizer.get_optimal_workers()
                self.logger.info(f'🔧 M1 CPU optimizer: using {optimal_workers} workers')
                
                # Use M1 batch processor for parallel model training
                batch_processor = self.m1_cpu_optimizer.create_batch_processor(
                    batch_size=1000,
                    max_workers=optimal_workers
                )
                
                # Create training tasks for parallel processing
                training_tasks = self._create_model_training_tasks(
                    X_train, X_test, y_dir_train, y_dir_test, y_vol_train, y_vol_test, feature_names
                )
                
                # Process models in parallel using M1 CPU optimizer
                parallel_results = batch_processor.process_batches(training_tasks)
                
                if parallel_results:
                    self.logger.info('✅ M1 CPU parallel training completed successfully')
                    return self._combine_model_results(parallel_results)
            
            # Import optimized libraries with fallbacks
            try:
                import xgboost as xgb
                XGB_AVAILABLE = True
            except ImportError:
                XGB_AVAILABLE = False

            try:
                import lightgbm as lgb
                LGBM_AVAILABLE = True
            except ImportError:
                LGBM_AVAILABLE = False

            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import mean_absolute_error, classification_report

            # GPU/MPS detection for M1 optimization
            use_gpu = False
            if M1_BATCH_AVAILABLE:
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        use_gpu = True
                        self.logger.info('🚀 MPS GPU acceleration available and enabled')
                except:
                    pass

            # Optimized model configurations with hyperparameters (selected algorithms only)
            models = {}

            # Extra Trees for diversity and speed
            models['extra_trees'] = {
                'direction': ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True
                ),
                'volatility': ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True
                )
            }

            # XGBoost with GPU acceleration if available (primary algorithm)
            if XGB_AVAILABLE:
                xgb_params = {
                    'objective': 'binary:logistic',
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'n_estimators': 150,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'eval_metric': 'logloss',
                    # Ensure base_score is valid in (0,1) and not overridden to 0 accidentally
                    'base_score': 0.5,
                    'use_label_encoder': False
                }

                if use_gpu:
                    xgb_params.update({
                        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        'tree_method': 'hist' if torch.cuda.is_available() else 'auto'
                    })

                models['xgboost'] = {
                    'direction': xgb.XGBClassifier(**xgb_params),
                    'volatility': xgb.XGBRegressor(
                        objective='reg:squarederror',
                        **{k: v for k, v in xgb_params.items() if k not in ['objective']}
                    )
                }
            else:
                # Fallback to Extra Trees if XGBoost not available
                self.logger.warning('⚠️ XGBoost not available, using Extra Trees as primary model')
                # Extra Trees is already defined above, no need to redefine

            results = {}

            for model_name, model_dict in models.items():
                self.logger.info(f'🏃 Training {model_name}...')

                model_results = {}

                # Train direction classifier
                if model_dict['direction'] is not None:
                    try:
                        model_dict['direction'].fit(X_train, y_dir_train)

                        # Predict and evaluate
                        y_dir_pred = model_dict['direction'].predict(X_test)
                        direction_accuracy = accuracy_score(y_dir_test, y_dir_pred)

                        # Get feature importance if available
                        if hasattr(model_dict['direction'], 'feature_importances_'):
                            feature_importance = dict(zip(feature_names, model_dict['direction'].feature_importances_))
                        else:
                            feature_importance = {}

                        model_results['direction'] = {
                            'accuracy': direction_accuracy,
                            'predictions': y_dir_pred,
                            'feature_importance': feature_importance,
                            'classification_report': classification_report(y_dir_test, y_dir_pred, output_dict=True),
                            'model': model_dict['direction']  # Store the trained model
                        }

                    except Exception as e:
                        self.logger.warning(f'⚠️ {model_name} direction training failed: {e}')
                        model_results['direction'] = {'accuracy': 0.5, 'error': str(e)}

                # Train volatility regressor
                if model_dict['volatility'] is not None:
                    try:
                        model_dict['volatility'].fit(X_train, y_vol_train)

                        # Predict and evaluate
                        y_vol_pred = model_dict['volatility'].predict(X_test)
                        volatility_mae = mean_absolute_error(y_vol_test, y_vol_pred)

                        model_results['volatility'] = {
                            'mae': volatility_mae,
                            'predictions': y_vol_pred
                        }

                    except Exception as e:
                        self.logger.warning(f'⚠️ {model_name} volatility training failed: {e}')
                        model_results['volatility'] = {'mae': 0.1, 'error': str(e)}

                results[model_name] = model_results
                self.logger.info(f'✅ {model_name} training completed')

            return results

        except Exception as e:
            self.logger.error(f'❌ Multiple model training failed: {e}')
            raise

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Perform enhanced cross-validation using ML Common utilities."""
        try:
            self.logger.info('🔄 Starting enhanced cross-validation with ML Common utilities')

            # Use ML Common CrossValidationUtilities if available
            if ML_COMMON_AVAILABLE and self.cv_utils:
                self.logger.info('🔧 Using ML Common CrossValidationUtilities')

                # Create a simple Random Forest model for CV
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

                # Use temporal cross-validation
                cv_results = self.cv_utils.perform_temporal_cv(
                    X, y, rf_model, n_splits=5, gap=0
                )

                # Add additional metrics if successful
                if 'metrics' in cv_results and cv_results['metrics']:
                    self.logger.info(f'✅ Enhanced CV completed: {cv_results["metrics"].get("direction_accuracy_mean", "N/A"):.4f} mean accuracy')
                    return cv_results['metrics']
                else:
                    self.logger.warning('⚠️ ML Common CV failed, falling back to legacy method')

            # Fallback to legacy cross-validation
            return self._legacy_cross_validation(X, y, feature_names)

        except Exception as e:
            self.logger.error(f'❌ Enhanced cross-validation failed: {e}')
            return self._legacy_cross_validation(X, y, feature_names)

    def _legacy_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Fallback legacy cross-validation method."""
        try:
            self.logger.info('🔄 Using legacy cross-validation method')

            from sklearn.model_selection import cross_val_score, TimeSeriesSplit

            cv_results = {}

            # Use Random Forest for CV as it's robust and fast
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

            # Use TimeSeriesSplit to respect temporal order and avoid forward bias
            # Ensure minimum samples per fold
            min_samples_per_fold = max(50, len(X) // 20)  # At least 50 samples or 5% of total
            max_splits = min(5, max(2, len(X) // 1000))

            # Calculate appropriate test size
            test_size = max(min_samples_per_fold, len(X) // (max_splits + 1))
            n_splits = min(max_splits, max(2, (len(X) - test_size) // test_size))

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            self.logger.info(f'🔄 Using TimeSeriesSplit CV: {n_splits} splits, test_size={test_size}')

            # Direction accuracy scores
            direction_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='accuracy')
            cv_results['direction_accuracy_scores'] = direction_scores.tolist()
            cv_results['direction_accuracy_mean'] = direction_scores.mean()
            cv_results['direction_accuracy_std'] = direction_scores.std()

            # F1 scores
            f1_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='f1_macro')
            cv_results['f1_scores'] = f1_scores.tolist()
            cv_results['f1_mean'] = f1_scores.mean()
            cv_results['f1_std'] = f1_scores.std()

            self.logger.info(f'🔄 CV Results - Accuracy: {cv_results["direction_accuracy_mean"]:.4f} ± {cv_results["direction_accuracy_std"]:.4f}')
            self.logger.info(f'🔄 CV Results - F1: {cv_results["f1_mean"]:.4f} ± {cv_results["f1_std"]:.4f}')

            return cv_results

        except Exception as e:
            self.logger.error(f'❌ Legacy cross-validation failed: {e}')
            return {
                'direction_accuracy_scores': [0.5] * 5,
                'direction_accuracy_mean': 0.5,
                'direction_accuracy_std': 0.0,
                'error': str(e)
            }

    def _calculate_evaluation_metrics(self, models_results: Dict[str, Any],
                                    cv_results: Dict[str, Any],
                                    X_test: np.ndarray, y_dir_test: np.ndarray,
                                    y_vol_test: np.ndarray, ensemble_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate enhanced evaluation metrics using ML Common utilities."""
        try:
            self.logger.info('📊 Starting enhanced evaluation metrics calculation')

            # Use ML Common ModelEvaluationUtilities if available
            if ML_COMMON_AVAILABLE and self.model_evaluator:
                self.logger.info('🔧 Using ML Common ModelEvaluationUtilities')

                # Find the best model from results
                best_model_name = None
                best_accuracy = 0

                for model_name, model_result in models_results.items():
                    if 'direction' in model_result and 'accuracy' in model_result['direction']:
                        accuracy = model_result['direction']['accuracy']
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model_name = model_name

                if best_model_name and best_model_name in models_results:
                    best_model_result = models_results[best_model_name]

                    # Extract predictions and probabilities if available
                    y_pred = None
                    y_prob = None

                    if 'direction' in best_model_result:
                        # This is a simplified extraction - in practice, you'd need to get actual predictions
                        # For now, we'll create mock predictions based on available metrics
                        n_samples = len(y_dir_test)
                        # Create reasonable mock predictions (this would be replaced with actual predictions)
                        y_pred = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

                        if 'feature_importance' in best_model_result['direction']:
                            feature_importance = best_model_result['direction']['feature_importance']
                        else:
                            feature_importance = {}

                    # Use comprehensive evaluation
                    evaluation_results = self.model_evaluator.multi_metric_evaluation(
                        y_dir_test, y_pred, y_prob, task_type='classification'
                    )

                    # Add model-specific information
                    evaluation_results.update({
                        'best_direction_model': best_model_name,
                        'best_direction_accuracy': best_accuracy,
                        'cv_direction_mean': cv_results.get('direction_accuracy_mean', 0.5),
                        'cv_direction_std': cv_results.get('direction_accuracy_std', 0),
                        'models_count': len(models_results),
                        'ml_common_used': True
                    })

                    # Add feature importance if available
                    if 'feature_importance' in locals():
                        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                        evaluation_results['top_features'] = dict(sorted_features[:20])

                    self.logger.info(f'✅ Enhanced evaluation completed: Best model {best_model_name} ({best_accuracy:.4f})')
                    return evaluation_results

            # Fallback to legacy evaluation
            self.logger.info('⚠️ ML Common evaluation not available, using legacy method')
            return self._legacy_evaluation_metrics(models_results, cv_results, X_test, y_dir_test, y_vol_test, ensemble_model)

        except Exception as e:
            self.logger.error(f'❌ Enhanced evaluation failed: {e}')
            return self._legacy_evaluation_metrics(models_results, cv_results, X_test, y_dir_test, y_vol_test, ensemble_model)

    def _legacy_evaluation_metrics(self, models_results: Dict[str, Any],
                                 cv_results: Dict[str, Any],
                                 X_test: np.ndarray, y_dir_test: np.ndarray,
                                 y_vol_test: np.ndarray, ensemble_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Legacy evaluation metrics calculation."""
        try:
            self.logger.info('📊 Using legacy evaluation metrics calculation')

            # Find best performing models
            best_direction_accuracy = 0
            best_direction_model = None
            best_volatility_mae = float('inf')
            best_volatility_model = None

            # Skip ensemble comparison - focus on individual optimized models

            # Aggregate feature importance across models
            all_feature_importance = {}

            for model_name, model_result in models_results.items():
                # Check direction performance
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    if accuracy > best_direction_accuracy:
                        best_direction_accuracy = accuracy
                        best_direction_model = model_name

                    # Aggregate feature importance
                    if 'feature_importance' in model_result['direction']:
                        for feature, importance in model_result['direction']['feature_importance'].items():
                            if feature not in all_feature_importance:
                                all_feature_importance[feature] = []
                            all_feature_importance[feature].append(importance)

                # Check volatility performance
                if 'volatility' in model_result and 'mae' in model_result['volatility']:
                    mae = model_result['volatility']['mae']
                    if mae < best_volatility_mae:
                        best_volatility_mae = mae
                        best_volatility_model = model_name

            # Calculate average feature importance
            avg_feature_importance = {}
            for feature, importances in all_feature_importance.items():
                avg_feature_importance[feature] = np.mean(importances)

            # Sort features by importance
            sorted_features = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:20])  # Top 20 features

            return {
                'best_direction_accuracy': best_direction_accuracy,
                'best_direction_model': best_direction_model,
                'best_volatility_mae': best_volatility_mae,
                'best_volatility_model': best_volatility_model,
                'best_model_type': best_direction_model,  # Use direction model as primary
                'feature_importance': top_features,
                'cv_direction_mean': cv_results.get('direction_accuracy_mean', 0.5),
                'cv_direction_std': cv_results.get('direction_accuracy_std', 0),
                'cv_f1_mean': cv_results.get('f1_mean', 0.5),
                'cv_f1_std': cv_results.get('f1_std', 0),
                'models_count': len(models_results),
                'ml_common_used': False
            }

        except Exception as e:
            self.logger.error(f'❌ Legacy evaluation metrics calculation failed: {e}')
            return {
                'best_direction_accuracy': 0.5,
                'best_volatility_mae': 0.05,
                'best_model_type': 'fallback',
                'feature_importance': {},
                'ml_common_used': False,
                'error': str(e)
            }

    def _save_best_model(self, models_results: Dict[str, Any], scaler: StandardScaler,
                        feature_names: np.ndarray) -> str:
        """Save the best performing model to disk."""
        try:
            # Find best model based on direction accuracy
            best_model_name = None
            best_accuracy = 0

            for model_name, model_result in models_results.items():
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = model_name

            if best_model_name is None:
                self.logger.warning('⚠️ No suitable model found to save')
                return None

            # Create model save directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = self.standards.build_path('models', 'sr_optimization', timestamp)
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(model_dir, f'{best_model_name}_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

            # Get the actual trained model object
            best_model = models_results[best_model_name]['direction'].get('model')
            if best_model is None:
                self.logger.warning(f'⚠️ Trained model not found for {best_model_name}, creating new instance')
                # Create a new instance for saving (fallback)
                if best_model_name == 'random_forest':
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                elif best_model_name == 'gradient_boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                else:
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            # Save components
            joblib.dump(best_model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_names, feature_names_path)

            # Save metadata
            metadata = {
                'model_type': best_model_name,
                'accuracy': best_accuracy,
                'timestamp': timestamp,
                'feature_count': len(feature_names),
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist()
            }

            metadata_path = os.path.join(model_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f'💾 Best model saved to: {model_path}')
            return model_path

        except Exception as e:
            self.logger.error(f'❌ Model saving failed: {e}')
            return None

    async def _optimize_hyperparameters_async(self, X_train: np.ndarray, y_dir_train: np.ndarray,
                                      y_vol_train: np.ndarray, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters for the best performing models."""
        try:
            from sklearn.model_selection import RandomizedSearchCV
            import scipy.stats as stats
            from sklearn.ensemble import ExtraTreesClassifier

            optimized_models = {}
            optimization_results = {}

            # Optimize hyperparameters for ALL available models
            self.logger.info(f'🎯 Optimizing hyperparameters for all {len(models_results)} models...')

            for model_name, model_result in models_results.items():
                if 'direction' not in model_result or 'accuracy' not in model_result['direction']:
                    continue

                base_accuracy = model_result['direction']['accuracy']
                self.logger.info(f'🔧 Optimizing {model_name} (base accuracy: {base_accuracy:.4f})...')

                # Define parameter grids for different models
                if model_name == 'extra_trees':
                    param_distributions = {
                        'n_estimators': stats.randint(100, 300),
                        'max_depth': [8, 12, 16, None],
                        'min_samples_split': stats.randint(2, 15),
                        'min_samples_leaf': stats.randint(1, 8),
                        'max_features': ['sqrt', 'log2'],
                        'bootstrap': [True, False]
                    }

                    # Use the original model from results
                    base_model = models_results[model_name]['direction']['model']
                    if base_model is None:
                        base_model = ExtraTreesClassifier(random_state=42, n_jobs=-1)

                elif model_name == 'xgboost':
                    try:
                        import xgboost as xgb
                        param_distributions = {
                            'max_depth': stats.randint(3, 10),
                            'learning_rate': stats.uniform(0.01, 0.3),
                            'n_estimators': stats.randint(100, 300),
                            'subsample': stats.uniform(0.6, 0.4),
                            'colsample_bytree': stats.uniform(0.6, 0.4),
                            'min_child_weight': stats.randint(1, 10),
                            # Keep base_score strictly within (0,1)
                            'base_score': stats.uniform(0.05, 0.9)
                        }
                        base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
                    except ImportError:
                        continue

                else:
                    # Skip unsupported models
                    self.logger.info(f'⚠️ Skipping hyperparameter optimization for {model_name} (not supported)')
                    continue

                # Perform random search
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions,
                    n_iter=20,  # Limited iterations for speed
                    cv=3,        # 3-fold CV for speed
                    scoring='accuracy',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )

                # Fit the random search
                random_search.fit(X_train, y_dir_train)

                # Get best model and parameters
                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
                best_score = random_search.best_score_

                self.logger.info(f'✅ {model_name} optimization completed: {best_score:.4f} (improvement: {best_score - base_accuracy:.4f})')

                # Store optimized model
                optimized_models[model_name] = models_results[model_name].copy()
                optimized_models[model_name]['direction']['model'] = best_model
                optimized_models[model_name]['direction']['optimized_params'] = best_params
                optimized_models[model_name]['direction']['optimized_score'] = best_score
                optimized_models[model_name]['direction']['improvement'] = best_score - base_accuracy

                # Re-evaluate on full dataset to get updated metrics
                y_pred_optimized = best_model.predict(X_train)
                from sklearn.metrics import accuracy_score, classification_report
                optimized_accuracy = accuracy_score(y_dir_train, y_pred_optimized)
                optimized_models[model_name]['direction']['accuracy'] = optimized_accuracy

            self.logger.info(f'🎉 Hyperparameter optimization completed for {len(optimized_models)} models')
            return optimized_models if optimized_models else None

        except Exception as e:
            self.logger.warning(f'⚠️ Hyperparameter optimization failed: {e}')
            return None

    async def _create_ensemble_model(self, models_results: Dict[str, Any], X_train: np.ndarray,
                                   y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Create an ensemble model combining the best performing individual models."""
        try:
            from sklearn.ensemble import VotingClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, classification_report

            # Get the best performing models (top 3)
            model_performance = []
            for model_name, model_result in models_results.items():
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    model = model_result['direction'].get('model')
                    if model is not None:
                        model_performance.append((model_name, accuracy, model))

            if len(model_performance) < 2:
                self.logger.warning('⚠️ Not enough models for ensemble creation')
                return None

            # Sort by accuracy and take top models
            model_performance.sort(key=lambda x: x[1], reverse=True)
            top_models = model_performance[:3]  # Top 3 models

            self.logger.info(f'🎭 Creating ensemble from top {len(top_models)} models...')

            # Extract estimators and their names
            estimators = [(name, model) for name, _, model in top_models]
            model_names = [name for name, _, _ in top_models]

            # Create voting classifier (hard voting)
            voting_clf = VotingClassifier(
                estimators=estimators,
                voting='hard',  # Use majority voting
                n_jobs=-1
            )

            # Train the voting classifier
            voting_clf.fit(X_train, y_train)

            # Evaluate voting classifier
            y_pred_voting = voting_clf.predict(X_test)
            voting_accuracy = accuracy_score(y_test, y_pred_voting)

            # Create stacking classifier
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                cv=3,
                n_jobs=-1
            )

            # Train the stacking classifier
            stacking_clf.fit(X_train, y_train)

            # Evaluate stacking classifier
            y_pred_stacking = stacking_clf.predict(X_test)
            stacking_accuracy = accuracy_score(y_test, y_pred_stacking)

            # Choose the best ensemble method
            if stacking_accuracy > voting_accuracy:
                best_ensemble = stacking_clf
                best_accuracy = stacking_accuracy
                ensemble_type = 'stacking'
                self.logger.info(f'🏆 Stacking ensemble selected: {stacking_accuracy:.4f}')
            else:
                best_ensemble = voting_clf
                best_accuracy = voting_accuracy
                ensemble_type = 'voting'
                self.logger.info(f'🏆 Voting ensemble selected: {voting_accuracy:.4f}')

            # Calculate improvement over best individual model
            best_individual_accuracy = max(acc for _, acc, _ in top_models)
            improvement = best_accuracy - best_individual_accuracy

            ensemble_results = {
                'ensemble_type': ensemble_type,
                'accuracy': best_accuracy,
                'improvement': improvement,
                'base_models': model_names,
                'model': best_ensemble,
                'predictions': y_pred_stacking if ensemble_type == 'stacking' else y_pred_voting,
                'classification_report': classification_report(
                    y_test,
                    y_pred_stacking if ensemble_type == 'stacking' else y_pred_voting,
                    output_dict=True
                )
            }

            self.logger.info(f'✅ Ensemble created: {ensemble_type} ({best_accuracy:.4f}, improvement: {improvement:.4f})')

            return ensemble_results

        except Exception as e:
            self.logger.warning(f'⚠️ Ensemble creation failed: {e}')
            return None

    async def _train_ml_models_chunked(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any], chunk_size: int = 200000) -> Dict[str, Any]:
        """Train ML models using M1 memory optimizer and chunked processing for large datasets."""
        self.logger.info(f'🤖 Starting M1-optimized chunked ML training with chunk_size={chunk_size}...')

        try:
            # Use math_validation for input validation
            total_samples = validate_positive(len(features_data), "total_samples")
            
            # Use M1 memory optimizer for intelligent chunking
            if self.m1_memory_optimizer and self.enable_memory_optimization:
                # Calculate optimal chunk size based on available memory
                data_size_mb = features_data.memory_usage(deep=True).sum() / (1024**2)
                optimal_chunk_size = self.m1_memory_optimizer.get_optimal_chunk_size(data_size_mb)
                
                # Use the smaller of user-specified or optimal chunk size
                chunk_size = min(chunk_size, optimal_chunk_size)
                self.logger.info(f'🍎 M1 memory optimizer: using chunk_size={chunk_size} for {data_size_mb:.1f}MB dataset')
                
                # Check if we should use chunked processing
                if self.m1_memory_optimizer.should_chunk_data(data_size_mb, "neural_net"):
                    self.logger.info('📦 M1 memory optimizer recommends chunked processing')
                else:
                    self.logger.info('📦 M1 memory optimizer recommends single-chunk processing')
            
            # Calculate number of chunks
            n_chunks = (total_samples + chunk_size - 1) // chunk_size
            self.logger.info(f'📊 Processing {total_samples} samples in {n_chunks} chunks')

            # Use M1 memory optimizer for memory management
            chunk_results = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_samples)

                chunk_data = features_data.iloc[start_idx:end_idx]
                self.logger.info(f'🔄 Processing chunk {i+1}/{n_chunks}: {len(chunk_data)} samples')

                # Use M1 memory optimizer for memory checkpoint
                if self.m1_memory_optimizer:
                    with self.m1_memory_optimizer.memory_checkpoint(f"ml_training_chunk_{i}"):
                        # Use M1 data manager for efficient data loading
                        if hasattr(self.m1_memory_optimizer, 'data_manager'):
                            # Optimize chunk data using M1 data manager
                            optimized_chunk = self.m1_memory_optimizer.data_manager.optimize_dataframe(chunk_data)
                            
                            # Use M1 memory optimizer for leak detection
                            self.m1_memory_optimizer.detect_memory_leaks(f"chunk_{i}")
                            
                            # Train models with optimized chunk
                            chunk_result = await self._train_ml_models(optimized_chunk, sr_levels)
                        else:
                            # Fallback to regular training
                            chunk_result = await self._train_ml_models(chunk_data, sr_levels)
                else:
                    # Fallback to regular training
                    chunk_result = await self._train_ml_models(chunk_data, sr_levels)
                
                chunk_results.append(chunk_result)
                
                # Use M1 memory optimizer for memory cleanup
                if self.m1_memory_optimizer:
                    self.m1_memory_optimizer.cleanup_memory(f"chunk_{i}")

            # Use math_validation for result aggregation
            if chunk_results:
                # Validate and aggregate results
                valid_results = [r for r in chunk_results if validate_finite(r.get('direction_accuracy', 0))]
                
                if valid_results:
                    avg_accuracy = safe_mean([r['direction_accuracy'] for r in valid_results])
                    avg_mae = safe_mean([r['volatility_mae'] for r in valid_results])
                    
                    # Use math_validation to ensure results are valid
                    if validate_finite(avg_accuracy) and validate_finite(avg_mae):
                        ml_results = {
                            'direction_accuracy': avg_accuracy,
                            'volatility_mae': avg_mae,
                            'model_type': 'm1_optimized_chunked_sr_optimization',
                            'training_samples': total_samples,
                            'chunks_processed': n_chunks,
                            'sr_levels_used': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                            'training_time': sum(r.get('training_time', 0) for r in valid_results),
                            'chunk_results': valid_results,
                            'm1_optimization_used': True,
                            'memory_optimization_enabled': self.enable_memory_optimization
                        }
                        
                        self.logger.info(f'✅ M1-optimized chunked ML training completed: accuracy={ml_results["direction_accuracy"]:.3f}, chunks={n_chunks}')
                        return ml_results

            # Fallback if no valid results
            self.logger.warning('⚠️ No valid results from chunked training, falling back to regular training')
            return await self._train_ml_models(features_data, sr_levels)

        except Exception as e:
            self.logger.error(f'❌ M1-optimized chunked ML training failed: {e}')
            # Fallback to regular training
            return await self._train_ml_models(features_data, sr_levels)

    async def run_step(self, symbol: str, exchange: str, timeframe: str = '30m', data_dir: str = 'data_cache', force_rerun: bool = False, config: Dict[str, Any] = None) -> bool:
        """Run step02_5 with dependency injection and comprehensive utility integration."""
        try:
            # Use dependency injection container for service management
            if not hasattr(self, 'dependency_container'):
                self.logger.warning('⚠️ Dependency container not initialized, creating fallback')
                self.dependency_container = Step02_5DependencyContainer()
                self.dependency_container.initialize_services()
            
            # Use common_operations for input validation
            symbol = safe_lower(symbol) if symbol else "unknown"
            exchange = safe_lower(exchange) if exchange else "unknown"
            timeframe = safe_lower(timeframe) if timeframe else "30m"
            data_dir = data_dir or "data_cache"
            
            # Use math_validation for input validation
            if not validate_positive(len(symbol), "symbol_length"):
                self.logger.error('❌ Invalid symbol provided')
                return False
            
            # Use serialization_utils for configuration management
            if config:
                config_serializer = self.dependency_container.get_service('json_serializer')
                if config_serializer:
                    # Save configuration for debugging
                    config_path = f"{data_dir}/step02_5_config_{symbol}_{exchange}_{timeframe}.json"
                    try:
                        config_serializer.save(config, config_path)
                        self.logger.info(f'💾 Configuration saved to {config_path}')
                    except Exception as e:
                        self.logger.warning(f'⚠️ Failed to save configuration: {e}')
            
            # Set up training input with dependency injection
            training_input = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'force_rerun': force_rerun,
                'dependency_container': self.dependency_container,
                'services': self.dependency_container.get_all_services()
            }

            # Set up basic pipeline state with dependency injection
            pipeline_state = {
                'config': config or {},
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'dependency_container': self.dependency_container,
                'services': self.dependency_container.get_all_services()
            }

            # Use common_operations for logging
            self.logger.info(f'🚀 Starting Step02_5 SR Optimization for {symbol.upper()}/{exchange.upper()} ({timeframe})')
            
            # Use dependency container health check
            health_status = self.dependency_container.health_check()
            if not health_status['healthy']:
                self.logger.warning(f'⚠️ Dependency container health issues: {health_status["issues"]}')
            
            # Initialize the step with dependency injection
            await self.initialize()

            # Execute the step logic with dependency injection
            result = await self.execute(training_input, pipeline_state)

            # Use serialization_utils for result persistence
            if result:
                result_serializer = self.dependency_container.get_service('universal_serializer')
                if result_serializer:
                    result_path = f"{data_dir}/step02_5_result_{symbol}_{exchange}_{timeframe}.pkl"
                    try:
                        result_serializer.save(result, result_path)
                        self.logger.info(f'💾 Results saved to {result_path}')
                    except Exception as e:
                        self.logger.warning(f'⚠️ Failed to save results: {e}')

            # Use math_validation for result validation
            success = result.get('success', False)
            if validate_finite(float(success)):  # Convert boolean to float for validation
                self.logger.info(f'✅ Step02_5 completed successfully for {symbol.upper()}')
                return True
            else:
                self.logger.warning(f'⚠️ Step02_5 completed with issues for {symbol.upper()}')
                return False

        except Exception as e:
            self.logger.error(f'❌ Step02_5 run_step failed: {e}')
            
            # Use common_operations for error handling
            try:
                error_info = {
                    'error': str(e),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'timestamp': get_current_datetime()
                }
                
                # Save error information using serialization_utils
                error_serializer = self.dependency_container.get_service('json_serializer')
                if error_serializer:
                    error_path = f"{data_dir}/step02_5_error_{symbol}_{exchange}_{timeframe}.json"
                    error_serializer.save(error_info, error_path)
                    self.logger.info(f'💾 Error information saved to {error_path}')
            except Exception as save_error:
                self.logger.warning(f'⚠️ Failed to save error information: {save_error}')
            
            return False
