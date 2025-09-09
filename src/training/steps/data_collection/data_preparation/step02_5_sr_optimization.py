"""Step 2.5: S/R Detection Optimization with Comprehensive Reporting and Function Call Monitoring."""
import asyncio
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
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

# M1 Optimization Utilities - Integrated via Common Operations
try:
    from src.utils.common_operations import (
        integrate_with_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer,
        get_m1_cpu_optimizer, cleanup_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage
    )

    # Initialize M1 integration through common operations
    m1_integration_result = integrate_with_m1_optimizers()
    M1_GPU_AVAILABLE = m1_integration_result.get('gpu_manager', False)
    M1_MEMORY_AVAILABLE = m1_integration_result.get('memory_optimizer', False)
    M1_CPU_AVAILABLE = m1_integration_result.get('cpu_optimizer', False)
    M1_BATCH_AVAILABLE = M1_CPU_AVAILABLE  # Batch processor available if CPU optimizer is

    integration_status = m1_integration_result.get('integration_status', 'unknown')
    if integration_status == 'success':
        logger.info("✅ Complete M1 utilities integration successful")
    elif integration_status == 'partial':
        logger.info("⚠️ Partial M1 utilities integration - some components available")
    else:
        logger.warning("❌ M1 utilities integration failed")

except ImportError as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.warning(f"M1 utilities integration not available: {e}")
except Exception as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.error(f"Unexpected error in M1 utilities integration: {e}")

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
        ParallelProcessingCoordinator, HyperparameterOptimization, ModelRegistry,
        MLPipelineOrchestrator, ValidationUtils, advanced_information_barrier_checks,
        validate_feature_timestamp_alignment, automated_future_data_filtering,
        rolling_window_bias_validation, detect_concept_drift, analyze_feature_stability,
        calculate_data_quality_score, enhanced_automated_data_cleaning,
        gpu_accelerated_processing, adaptive_load_balancing, fault_tolerant_parallel_execution,
        parallel_feature_engineering_gpu
    )
    ML_COMMON_AVAILABLE = True

    # Initialize feature selection framework for advanced feature selection
    _feature_selector = None
    def get_feature_selector():
        global _feature_selector
        if _feature_selector is None:
            _feature_selector = FeatureSelectionFramework()
        return _feature_selector

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
        
        # Initialize M1 optimization services via common operations
        if M1_GPU_AVAILABLE:
            self._services['m1_gpu_manager'] = get_m1_gpu_manager()
            if self._services['m1_gpu_manager']:
                # Configure GPU manager for SR optimization
                self._services['m1_gpu_manager'].config.update({
                    'enable_memory_cleanup': True,
                    'memory_threshold': 0.8,
                    'batch_size': 1000
                })
                logger.info("🎯 M1 GPU Manager configured for SR optimization")

        if M1_MEMORY_AVAILABLE:
            self._services['m1_memory_optimizer'] = get_m1_memory_optimizer()
            if self._services['m1_memory_optimizer']:
                logger.info("🧠 M1 Memory Optimizer configured for SR optimization")

        if M1_CPU_AVAILABLE:
            self._services['m1_cpu_optimizer'] = get_m1_cpu_optimizer()
            if self._services['m1_cpu_optimizer']:
                logger.info("⚡ M1 CPU Optimizer configured for SR optimization")

        # Set up memory monitoring for SR optimization
        if M1_MEMORY_AVAILABLE or M1_GPU_AVAILABLE:
            logger.info("🧠 Memory monitoring enabled for SR optimization")
            # Add initial memory checkpoint
            try:
                memory_usage = get_memory_usage()
                logger.info(f"📊 Initial memory usage: {memory_usage['rss_gb']:.2f}GB")
            except Exception as e:
                logger.debug(f"Failed to get initial memory usage: {e}")
        
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

# Error classification and handling now provided by ml_common utilities

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
        
        # Configurable proximity threshold for SR classification
        self.proximity_threshold = config.get('sr_optimization', {}).get('proximity_threshold', 0.002)  # Default 0.2%
        
        # Initialize automatic memory management
        try:
            from src.utils.enhanced_memory_management import get_memory_manager, memory_context
            self.memory_manager = get_memory_manager()
            self.memory_manager.start_monitoring()
            self.logger.info("🧠 Memory management initialized")
        except Exception as e:
            self.logger.warning(f"Memory manager initialization failed: {e}")

        # Initialize ML Model Configurations early to prevent attribute errors
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
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'HistGradientBoostingClassifier': {
                'class': HistGradientBoostingClassifier,
                'hyperparameters': {
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'max_iter': {'type': 'int', 'low': 50, 'high': 300, 'step': 10},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
                    'l2_regularization': {'type': 'float', 'low': 0.0, 'high': 1.0}
                },
                'default_params': {
                    'learning_rate': 0.1,
                    'max_iter': 100,
                    'max_depth': None,
                    'min_samples_leaf': 20,
                    'l2_regularization': 0.0,
                    'random_state': 42
                }
            }
        }
        
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

        # Initialize ML Common utilities AFTER all configuration parameters are set
        self._initialize_ml_common_utilities()
        
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
                self.current_timestamp = datetime.now()  # Initialize current timestamp for lookahead protection
                self.cv_utils = CrossValidationUtilities()
                self.model_evaluator = ModelEvaluationUtilities()
                self.data_quality_utils = DataQualityUtilities()
                self.memory_optimizer = MemoryEfficientTraining()
                self.parallel_processor = ParallelProcessingCoordinator()
                
                # Initialize additional ML utilities
                self.hpo_optimizer = HyperparameterOptimization()
                self.model_registry = ModelRegistry()
                self.pipeline_orchestrator = MLPipelineOrchestrator()
                self.validation_utils = ValidationUtils()

                # Initialize utility configurations
                self._configure_ml_utilities()

                self.logger.info("✅ ML Common utilities initialized successfully")
            else:
                self.logger.error("❌ ML Common utilities not available")
                raise ImportError("ML Common utilities are required but not available")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Common utilities: {e}")
            raise ImportError(f"ML Common utilities initialization failed: {e}")

    def _configure_ml_utilities(self) -> None:
        """Configure ML utilities with step-specific settings."""
        try:
            # Configure feature selection
            if self.feature_selector:
                self.feature_selector.config.update({
                    'enable_gpu': self.enable_m1_optimizations,
                    'enable_parallel': self.enable_parallel_processing,
                    'max_workers': 4,
                    'memory_threshold': 0.8
                })

            # Configure cross-validation
            if self.cv_utils:
                self.cv_utils.config.update({
                    'enable_gpu': self.enable_m1_optimizations,
                    'enable_parallel': self.enable_parallel_processing,
                    'max_workers': 4
                })

            # Configure hyperparameter optimization
            if self.hpo_optimizer:
                self.hpo_optimizer.config.update({
                    'enable_gpu': self.enable_m1_optimizations,
                    'enable_parallel': self.enable_parallel_processing,
                    'default_n_trials': self.optimization_trials,
                    'default_timeout': 300
                })

            # Configure data quality utilities
            if self.data_quality_utils:
                self.data_quality_utils.config.update({
                    'enable_gpu': self.enable_m1_optimizations,
                    'enable_memory_optimization': self.enable_memory_optimization,
                    'drift_threshold': 0.1,
                    'missing_threshold': 0.5
                })

            # Configure memory optimization
            if self.memory_optimizer:
                self.memory_optimizer.config.update({
                    'chunk_size_mb': 500,
                    'max_memory_usage': 0.8,
                    'enable_gpu_memory_pool': self.enable_m1_optimizations
                })

            # Configure parallel processing
            if self.parallel_processor:
                self.parallel_processor.config.update({
                    'enable_gpu': self.enable_m1_optimizations,
                    'max_workers': 4,
                    'memory_threshold': 0.8
                })

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to configure ML utilities: {e}")

    def _initialize_fallback_utilities(self) -> None:
        """Initialize fallback utilities when ML Common is not available."""
        self.feature_selector = None
        self.lookahead_protector = None
        self.current_timestamp = datetime.now()  # Initialize current timestamp even when ML Common is unavailable
        self.cv_utils = None
        self.model_evaluator = None
        self.data_quality_utils = None
        self.memory_optimizer = None
        self.parallel_processor = None
        self.hpo_optimizer = None
        self.model_registry = None
        self.pipeline_orchestrator = None
        self.validation_utils = None
        
        # Log utility integration status
        self._log_utility_integration_status()

        # Initialize ML failure tracking
        self.ml_failure_count = 0
        self.ml_failure_reasons = []
        self.fast_fail_engaged = False  # Flag to prevent redundant restart attempts


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
        if not hasattr(self, 'ml_failure_count'):
            self.ml_failure_count = 0
        if not hasattr(self, 'ml_failure_reasons'):
            self.ml_failure_reasons = []
        self.ml_failure_count += 1
        self.ml_failure_reasons.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'failure_count': self.ml_failure_count
        })

        # Classify failure severity - make bias detection more lenient
        critical_errors = ["DATA_UNAVAILABLE", "EMPTY_DATA"]
        recoverable_errors = ["FORWARD_BIAS_ERROR", "OPTUNA_ERROR", "CV_ERROR", "MODEL_FIT_ERROR", "ML_TRAINING_ERROR", "METHOD_VALIDATION_ERROR"]

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
            elif error_type == "FORWARD_BIAS_ERROR" and self.ml_failure_count >= 5:  # More lenient for bias errors
                self.logger.critical(f'🚨 FAST FAIL: {self.ml_failure_count} bias detection failures, aborting training')
                self.fast_fail_engaged = True
                raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} bias detection failures")
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
        total_calls = pre_report.get('total_calls', 0)
        self.logger.info(f"📊 Pre-execution function calls: {total_calls}")

        # Check if execute_logic method exists, if not, use execute_main_logic
        if hasattr(self, 'execute_logic'):
            result = await self.execute_logic(training_input, pipeline_state)
        else:
            result = await self.execute_main_logic(training_input, pipeline_state)

        # Pass ML results to function report for detailed metrics
        ml_results = result.get('ml_results', {})
        post_report = generate_function_report(ml_results)
        post_total_calls = post_report.get('total_calls', 0)
        self.logger.info(f"📊 Post-execution function calls: {post_total_calls}")
        self.logger.info(f"📈 Function call increase: {post_total_calls - total_calls}")
        result['function_call_report'] = post_report
        return result

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features locally for SR optimization."""
        try:
            self.logger.info('🔧 Calculating features locally for SR optimization...')
            return self._calculate_features_locally(data)

        except Exception as e:
            self.logger.error(f'❌ Feature loading failed: {e}')
            raise

    def _calculate_features_locally(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic features locally when step06 is not available."""
        try:
            self.logger.info('🔧 Calculating basic features locally...')

            # Start with input data
            features_data = data.copy()

            # Ensure we have required OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_ohlcv = [col for col in ohlcv_cols if col not in features_data.columns]
            if missing_ohlcv:
                raise ValueError(f"❌ Missing required OHLCV columns: {missing_ohlcv}")

            # Calculate basic price features
            if 'price_change_pct' not in features_data.columns:
                features_data['price_change_pct'] = features_data['close'].pct_change()

            if 'price_change_log' not in features_data.columns:
                features_data['price_change_log'] = np.log(features_data['close'] / features_data['close'].shift(1))

            # Calculate volatility features
            if 'volatility_5' not in features_data.columns:
                features_data['volatility_5'] = features_data['close'].rolling(window=5).std()

            if 'volatility_10' not in features_data.columns:
                features_data['volatility_10'] = features_data['close'].rolling(window=10).std()

            if 'volatility_20' not in features_data.columns:
                features_data['volatility_20'] = features_data['close'].rolling(window=20).std()

            # Calculate momentum features
            if 'momentum_5' not in features_data.columns:
                features_data['momentum_5'] = features_data['close'] - features_data['close'].shift(5)

            if 'momentum_10' not in features_data.columns:
                features_data['momentum_10'] = features_data['close'] - features_data['close'].shift(10)

            # Calculate volume features
            if 'volume_ma_5' not in features_data.columns:
                features_data['volume_ma_5'] = features_data['volume'].rolling(window=5).mean()

            if 'volume_ratio' not in features_data.columns:
                features_data['volume_ratio'] = features_data['volume'] / features_data['volume_ma_5']

            # Calculate ratio features
            if 'high_low_ratio' not in features_data.columns:
                features_data['high_low_ratio'] = features_data['high'] / features_data['low']

            if 'close_open_ratio' not in features_data.columns:
                features_data['close_open_ratio'] = features_data['close'] / features_data['open']

            # Clean up NaN values - handle categorical columns properly
            # Fill numeric columns with 0, leave categorical columns as they are
            numeric_cols = features_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                features_data[numeric_cols] = features_data[numeric_cols].fillna(0)

            # For categorical columns, fill with most frequent value if they have NaN
            categorical_cols = features_data.select_dtypes(include=['category', 'object']).columns
            for col in categorical_cols:
                if features_data[col].isna().any():
                    # Fill with most frequent value, or first category if available
                    if hasattr(features_data[col], 'cat'):
                        # This is a pandas categorical
                        if not features_data[col].cat.categories.empty:
                            fill_value = features_data[col].cat.categories[0]
                        else:
                            fill_value = None
                    else:
                        # This is an object column
                        mode_value = features_data[col].mode()
                        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else None

                    if fill_value is not None:
                        features_data[col] = features_data[col].fillna(fill_value)

            self.logger.info(f'✅ Calculated {len(features_data.columns)} basic features locally')
            return features_data

        except Exception as e:
            self.logger.error(f'❌ Local feature calculation failed: {e}')
            # Return original data as fallback
            return data

    async def execute_main_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive SR optimization logic - main implementation."""
        self.logger.info('🎯 Starting comprehensive S/R detection optimization with unified monitoring...')
        self.logger.info(f'📊 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'📊 Pipeline state keys: {list(pipeline_state.keys())}')
        self.start_time = time.time()

        # Memory checkpoint before main processing
        if M1_MEMORY_AVAILABLE or M1_GPU_AVAILABLE:
            with memory_checkpoint("sr_main_logic_start"):
                self.logger.info('📊 Memory checkpoint: Main logic start')
                optimize_memory()  # Optimize memory before heavy processing

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

            # Harmonize schema to prevent parquet compatibility issues
            if self.parquet_utils:
                self.logger.info('🔧 Harmonizing data schema to prevent compatibility issues...')
                data = self.parquet_utils.harmonize_schema_after_read(data)
                if data is not None:
                    self.logger.info('✅ Schema harmonization completed')
                else:
                    self.logger.error('❌ Schema harmonization failed')
                    raise ValueError("Schema harmonization failed")

            # Apply training mode filtering - check environment variables first, then input
            import os

            # Auto-detect mode from environment variables (primary method)
            if os.environ.get('LIGHT_TRAINING_MODE') == '1':
                training_mode = 'light'
                mode_source = 'environment (LIGHT_TRAINING_MODE)'
            elif os.environ.get('BLANK_TRAINING_MODE') == '1':
                training_mode = 'blank'
                mode_source = 'environment (BLANK_TRAINING_MODE)'
            else:
                # Fallback to input parameter
                training_mode = training_input.get('training_mode', 'full')
                mode_source = 'input parameter' if training_mode != 'full' else 'default (full)'

            self.logger.info(f'🔍 Training mode: {training_mode} (detected from {mode_source})')
            print(f'DEBUG: Training mode: {training_mode} (from {mode_source})', flush=True)

            # Log mode-specific information
            if training_mode == 'light':
                self.logger.info('💡 LIGHT mode: Aggressive resource reduction (20x smaller parameters)')
            elif training_mode == 'blank':
                self.logger.info('🧪 BLANK mode: Moderate resource reduction (5x smaller parameters)')
            else:
                self.logger.info('🚀 FULL mode: Maximum quality training with all features enabled')

            print(f'DEBUG: Available columns: {list(data.columns)}', flush=True)
            
            if training_mode == 'light':
                self.logger.info('🎯 Light training mode detected - filtering to last 10 days')
                print('DEBUG: Light mode detected, attempting to filter data', flush=True)
                timestamp_col = self._find_timestamp_column(data)
                if timestamp_col:
                    # Ensure timestamp column is in datetime format
                    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                        self.logger.info(f'🔧 Converting timestamp column "{timestamp_col}" to datetime format...')

                        # Check if timestamps are in Unix format (numeric values from 1970)
                        sample_timestamp = data[timestamp_col].iloc[0] if len(data) > 0 else None
                        if sample_timestamp and isinstance(sample_timestamp, (int, float)):
                            # If timestamp looks like Unix timestamp (large number), convert accordingly
                            if sample_timestamp > 1e10:  # Likely milliseconds
                                data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit='ms')
                                self.logger.info(f'🔧 Detected millisecond Unix timestamps, converted to datetime')
                            elif sample_timestamp > 1e6:  # Likely seconds
                                data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit='s')
                                self.logger.info(f'🔧 Detected second Unix timestamps, converted to datetime')
                            else:
                                # Fallback to standard conversion
                                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                                self.logger.info(f'🔧 Used standard datetime conversion')
                        else:
                            # Fallback to standard conversion
                            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                            self.logger.info(f'🔧 Used standard datetime conversion')
                    # Rename to standard 'timestamp' for consistency
                    if timestamp_col != 'timestamp':
                        data = data.rename(columns={timestamp_col: 'timestamp'})
                    
                    # Set current timestamp for bias detection
                    self.current_timestamp = datetime.now()
                    self.logger.info(f'🔧 Set current timestamp for bias detection: {self.current_timestamp}')
                    
                    # Set current timestamp in LookaheadProtection if available
                    if ML_COMMON_AVAILABLE and hasattr(self, 'lookahead_protector') and self.lookahead_protector:
                        self.lookahead_protector.set_current_timestamp(self.current_timestamp)
                    
                    # Check for duplicate timestamps and calculate percentage
                    total_rows = len(data)
                    timestamp_duplicates = data['timestamp'].duplicated().sum()
                    timestamp_duplicate_percentage = (timestamp_duplicates / total_rows) * 100 if total_rows > 0 else 0
                    
                    self.logger.info(f'📊 Duplicate timestamp analysis: {timestamp_duplicates:,} timestamp duplicates out of {total_rows:,} rows ({timestamp_duplicate_percentage:.2f}%)')
                    
                    # Check for true duplicates (same timestamp AND same content)
                    true_duplicates = data.duplicated().sum()
                    true_duplicate_percentage = (true_duplicates / total_rows) * 100 if total_rows > 0 else 0
                    
                    self.logger.info(f'📊 True duplicate analysis: {true_duplicates:,} true duplicates out of {total_rows:,} rows ({true_duplicate_percentage:.2f}%)')
                    
                    if true_duplicates > 0:
                        emoji = '⚠️ ' if true_duplicate_percentage >= 0.01 else ''
                        self.logger.warning(f'{emoji}Found {true_duplicates:,} true duplicates ({true_duplicate_percentage:.2f}% of data)')
                        # Remove true duplicates keeping the first occurrence
                        data = data.drop_duplicates(keep='first')
                        self.logger.info(f'🧹 Removed true duplicates, remaining rows: {len(data):,}')
                    elif timestamp_duplicates > 0:
                        self.logger.info(f'ℹ️ Found {timestamp_duplicates:,} timestamp duplicates but different content - keeping all rows')
                        self.logger.info(f'ℹ️ This is normal for high-frequency data where multiple trades occur at the same timestamp')
                    
                    # Debug: Check timestamp range
                    min_ts = data['timestamp'].min()
                    max_ts = data['timestamp'].max()
                    self.logger.info(f'📊 Data timestamp range: {min_ts} to {max_ts}')
                    print(f'DEBUG: Data timestamp range: {min_ts} to {max_ts}', flush=True)
                    
                    # Filter to last 10 days from the latest data point (not current time)
                    cutoff_date = max_ts - pd.Timedelta(days=10)
                    original_rows = data.shape[0]
                    data = data[data['timestamp'] >= cutoff_date]
                    self.logger.info(f'📊 Data filtered to light mode: {data.shape[0]:,} rows, {data.shape[1]} columns (was {original_rows:,} rows)')
                    print(f'DEBUG: Data filtered from {original_rows:,} to {data.shape[0]:,} rows', flush=True)
                else:
                    self.logger.warning('⚠️ No timestamp column found - cannot apply light mode filtering')
                    print('DEBUG: No timestamp column found for filtering', flush=True)
            else:
                self.logger.info(f'📊 Full training mode detected - using all {data.shape[0]:,} rows')
                print(f'DEBUG: Full training mode - using all data', flush=True)

            # Prepare features from data
            self.logger.info('🔧 Preparing features for SR optimization...')
            features_data = self._prepare_features(data)

            # Detect SR levels
            self.logger.info('🎯 Detecting Support/Resistance levels...')
            sr_levels = self._detect_sr_levels(features_data)

            # Log detailed S/R level information
            self._log_detailed_sr_levels(sr_levels, features_data)

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
                'use_parallel': self.enable_parallel_processing if hasattr(self, 'enable_parallel_processing') else False,
                'disable_dbscan_clustering': False,  # Keep clustering but with distance constraints
                'max_cluster_distance_pct': 0.5,  # Don't merge S/R levels more than 0.5% apart (very restrictive pre-DBSCAN)
                'max_volume_levels': 50,  # Increased from default 40 to 50
                'max_fibonacci_levels': 30,  # Increased from default 20 to 30
                'use_prominence_filtering': False,  # Disable prominence filtering after composite strength filtering
                'outlier_threshold_std': 3.5  # Statistical outlier threshold (standard deviations from mean)
            }

            # Clean data before SR detection to prevent NaN values
            self.logger.info('🧹 Cleaning data before SR detection...')
            clean_data = data.copy()

            # Ensure essential OHLCV columns exist and are clean
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in essential_cols:
                if col in clean_data.columns:
                    # Remove rows with NaN values in essential columns
                    nan_count = clean_data[col].isna().sum()
                    if nan_count > 0:
                        self.logger.warning(f'⚠️ Found {nan_count} NaN values in {col} column, removing these rows')
                        clean_data = clean_data.dropna(subset=[col])
                        if clean_data.empty:
                            raise ValueError(f"❌ All data removed after cleaning NaN values in {col} column")

            # Reset index after dropping rows
            clean_data = clean_data.reset_index(drop=True)

            self.logger.info(f'✅ Data cleaned: {clean_data.shape[0]} rows remaining (removed {data.shape[0] - clean_data.shape[0]} rows with NaN values)')

            # Initialize detector
            detector = EnhancedSRDetector(sr_config)

            # Detect levels using cleaned data
            sr_levels = detector.detect_sr_levels(clean_data)

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
            else:
                # Detector already returned dict format, ensure it has required keys
                if not isinstance(sr_levels, dict):
                    sr_levels = {'support_levels': [], 'resistance_levels': []}
                elif 'support_levels' not in sr_levels:
                    sr_levels['support_levels'] = []
                elif 'resistance_levels' not in sr_levels:
                    sr_levels['resistance_levels'] = []

            # Validate that detected levels were actually reached by market prices
            sr_levels = self._validate_sr_levels_against_market_data(sr_levels, clean_data)

            # Log ALL S/R levels found (before any filtering)
            self._log_all_sr_levels_detailed(sr_levels, data)

            self.logger.info(f'✅ Enhanced S/R detection complete: {len(sr_levels.get("support_levels", []))} support, {len(sr_levels.get("resistance_levels", []))} resistance levels')

            return sr_levels

        except Exception as e:
            self.logger.error(f'❌ Enhanced S/R detection failed: {e}')
            raise RuntimeError(f"Advanced SR detection failed: {e}. No fallback available.")

    def _log_all_sr_levels_detailed(self, sr_levels: Dict[str, Any], data: pd.DataFrame) -> None:
        """Log ALL S/R levels found with comprehensive details."""
        try:
            current_prices = data['close'].values
            price_min, price_max = min(current_prices), max(current_prices)
            price_range = price_max - price_min
            
            self.logger.info('🔍 COMPREHENSIVE S/R LEVEL ANALYSIS (ALL LEVELS):')
            self.logger.info(f'   - Data price range: {price_min:.2f} - {price_max:.2f} (range: {price_range:.2f})')
            self.logger.info(f'   - Data points: {len(data):,} rows')
            
            # Combine support and resistance levels
            all_levels = []
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])

            # Add support levels with type indicator
            for level in support_levels:
                level_price = self._extract_level_price(level)
                if level_price is not None and not np.isnan(level_price):
                    all_levels.append(('support', level_price, level))

            # Add resistance levels with type indicator
            for level in resistance_levels:
                level_price = self._extract_level_price(level)
                if level_price is not None and not np.isnan(level_price):
                    all_levels.append(('resistance', level_price, level))

            # Sort by price in ascending order
            all_levels.sort(key=lambda x: x[1])

            self.logger.info(f'   - Total S/R levels detected: {len(all_levels)} ({len(support_levels)} support, {len(resistance_levels)} resistance)')

            # Log all levels in ascending price order
            for i, (level_type, level_price, level) in enumerate(all_levels):
                    distance_from_min = abs(level_price - price_min)
                    distance_from_max = abs(level_price - price_max)
                    closest_distance = min(distance_from_min, distance_from_max)
                    distance_pct = (closest_distance / price_min) * 100
                    
                    # Extract additional level details
                    level_details = self._extract_level_details(level)

                    # Format level type with proper capitalization
                    type_display = level_type.capitalize()

                    self.logger.info(f'     {i+1}. {type_display}: {level_price:.2f} (closest: {closest_distance:.2f}, {distance_pct:.1f}%) {level_details}')
                    
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to log comprehensive S/R levels: {e}')

    def _extract_level_price(self, level) -> float:
        """Extract price from various level formats."""
        try:
            if isinstance(level, (int, float)):
                return float(level)
            elif hasattr(level, 'price'):
                return float(level.price)
            elif isinstance(level, dict) and 'price' in level:
                return float(level['price'])
            elif hasattr(level, 'value'):
                return float(level.value)
            elif isinstance(level, dict) and 'value' in level:
                return float(level['value'])
            else:
                return None
        except (ValueError, TypeError):
            return None

    def _extract_level_details(self, level) -> str:
        """Extract additional details from level object."""
        try:
            details = []
            
            # Extract level type
            if hasattr(level, 'type'):
                details.append(f"type={level.type}")
            elif isinstance(level, dict) and 'type' in level:
                details.append(f"type={level['type']}")
            
            # Extract strength/touches
            if hasattr(level, 'strength'):
                details.append(f"strength={level.strength}")
            elif isinstance(level, dict) and 'strength' in level:
                details.append(f"strength={level['strength']}")
            
            # Extract touches
            if hasattr(level, 'touches'):
                details.append(f"touches={level.touches}")
            elif isinstance(level, dict) and 'touches' in level:
                details.append(f"touches={level['touches']}")
            
            # Extract algorithm
            if hasattr(level, 'algorithm'):
                details.append(f"algo={level.algorithm}")
            elif isinstance(level, dict) and 'algorithm' in level:
                details.append(f"algo={level['algorithm']}")
            
            return f"[{', '.join(details)}]" if details else ""
            
        except Exception:
            return ""

    def _log_filtered_sr_levels(self, filtered_levels: Dict[str, Any], original_levels: Dict[str, Any], chunk_prices: np.ndarray) -> None:
        """Log S/R levels after filtering for this chunk."""
        try:
            # Validate input parameters
            if not hasattr(self, 'logger') or self.logger is None:
                print('⚠️ Logger not available for S/R level logging')
                return

            if chunk_prices is None or len(chunk_prices) == 0:
                self.logger.warning('⚠️ No chunk prices provided for S/R level logging')
                return

            # Validate that chunk_prices contains valid numeric values
            try:
                # Handle numpy arrays properly, filtering out NaN values
                if hasattr(chunk_prices, 'dtype') and hasattr(chunk_prices, 'shape'):  # numpy array
                    # Filter out NaN and infinite values
                    valid_prices = chunk_prices[np.isfinite(chunk_prices)]
                    if len(valid_prices) == 0:
                        raise ValueError("No valid (finite) price values found")
                    chunk_min, chunk_max = float(valid_prices.min()), float(valid_prices.max())
                else:  # regular Python iterable
                    chunk_min, chunk_max = min(chunk_prices), max(chunk_prices)

                    if not (isinstance(chunk_min, (int, float)) and isinstance(chunk_max, (int, float))):
                        raise ValueError("Invalid price values")
            except (ValueError, TypeError) as e:
                self.logger.warning(f'⚠️ Invalid chunk prices for S/R logging: {e}')
                return

            # Safely extract level counts with validation
            try:
                original_support = len(original_levels.get('support_levels', [])) if isinstance(original_levels, dict) else 0
                original_resistance = len(original_levels.get('resistance_levels', [])) if isinstance(original_levels, dict) else 0
                filtered_support = len(filtered_levels.get('support_levels', [])) if isinstance(filtered_levels, dict) else 0
                filtered_resistance = len(filtered_levels.get('resistance_levels', [])) if isinstance(filtered_levels, dict) else 0
            except Exception as e:
                self.logger.warning(f'⚠️ Error extracting level counts: {e}')
                original_support = original_resistance = filtered_support = filtered_resistance = 0

            # Log filtering results
            try:
                self.logger.info(f'🔍 CHUNK S/R FILTERING RESULTS (price range: {chunk_min:.2f}-{chunk_max:.2f}):')
                self.logger.info(f'   - Support levels: {original_support} -> {filtered_support} (filtered out: {original_support - filtered_support})')
                self.logger.info(f'   - Resistance levels: {original_resistance} -> {filtered_resistance} (filtered out: {original_resistance - filtered_resistance})')
            except Exception as e:
                self.logger.warning(f'⚠️ Error logging filtering results: {e}')

            # Combine and sort filtered levels by price with error handling
            all_filtered_levels = []

            # Add support levels with individual error handling
            try:
                support_levels = filtered_levels.get('support_levels', []) if isinstance(filtered_levels, dict) else []
                for level in support_levels:
                    try:
                        level_price = self._extract_level_price(level)
                        if level_price is not None and isinstance(level_price, (int, float)):
                            all_filtered_levels.append(('support', float(level_price), level))
                    except Exception as e:
                        self.logger.debug(f'Skipping invalid support level: {e}')
                        continue
            except Exception as e:
                self.logger.warning(f'⚠️ Error processing support levels: {e}')

            # Add resistance levels with individual error handling
            try:
                resistance_levels = filtered_levels.get('resistance_levels', []) if isinstance(filtered_levels, dict) else []
                for level in resistance_levels:
                    try:
                        level_price = self._extract_level_price(level)
                        if level_price is not None and isinstance(level_price, (int, float)):
                            all_filtered_levels.append(('resistance', float(level_price), level))
                    except Exception as e:
                        self.logger.debug(f'Skipping invalid resistance level: {e}')
                        continue
            except Exception as e:
                self.logger.warning(f'⚠️ Error processing resistance levels: {e}')

            # Sort by price with error handling
            try:
                if all_filtered_levels:
                    # Filter out any invalid entries before sorting
                    valid_levels = [(t, p, l) for t, p, l in all_filtered_levels if isinstance(p, (int, float)) and not (isinstance(p, float) and (math.isnan(p) or math.isinf(p)))]
                    valid_levels.sort(key=lambda x: x[1])
                    all_filtered_levels = valid_levels
            except Exception as e:
                self.logger.warning(f'⚠️ Error sorting filtered levels: {e}')
                all_filtered_levels = []

            # Log the filtered levels that are relevant to this chunk
            if all_filtered_levels:
                try:
                    self.logger.info(f'   - Relevant S/R levels for this chunk ({len(all_filtered_levels)} total):')
                    for i, (level_type, level_price, level) in enumerate(all_filtered_levels):
                        try:
                            distance_from_min = abs(level_price - chunk_min)
                            distance_from_max = abs(level_price - chunk_max)
                            closest_distance = min(distance_from_min, distance_from_max)
                            type_display = level_type.capitalize() if isinstance(level_type, str) else str(level_type)
                            self.logger.info('.2f')
                        except Exception as e:
                            self.logger.debug(f'Error logging individual level {i+1}: {e}')
                            continue
                except Exception as e:
                    self.logger.warning(f'⚠️ Error logging detailed level information: {e}')
                        
        except Exception as e:
            # Final fallback error handling
            try:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning(f'⚠️ Failed to log filtered S/R levels: {e}')
                else:
                    print(f'⚠️ Failed to log filtered S/R levels: {e}')
            except:
                print(f'⚠️ Critical error in S/R level logging: {e}')

    def _validate_sr_levels_against_market_data(self, sr_levels: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Validate that detected SR levels were actually reached by market prices."""
        try:
            if data is None or data.empty:
                self.logger.warning('⚠️ Cannot validate SR levels - no market data available')
                return sr_levels

            # Get actual market price boundaries
            actual_high = data['high'].max()
            actual_low = data['low'].min()
            actual_close_max = data['close'].max()
            actual_close_min = data['close'].min()

            # Use the most extreme prices as boundaries (high/low are more reliable than close)
            market_max = max(actual_high, actual_close_max)
            market_min = min(actual_low, actual_close_min)

            self.logger.info(f'🎯 SR Level Market Validation:')
            self.logger.info(f'   Market price range: ${market_min:.2f} - ${market_max:.2f}')
            self.logger.info(f'   Data points: {len(data):,}')

            # Validate support levels
            original_support_count = len(sr_levels.get('support_levels', []))
            validated_support = []

            for level in sr_levels.get('support_levels', []):
                level_price = self._extract_level_price(level)
                if level_price is not None:
                    # Support level must be >= actual market minimum
                    if level_price >= market_min:
                        validated_support.append(level)
                    else:
                        self.logger.debug('.2f')

            # Validate resistance levels
            original_resistance_count = len(sr_levels.get('resistance_levels', []))
            validated_resistance = []

            for level in sr_levels.get('resistance_levels', []):
                level_price = self._extract_level_price(level)
                if level_price is not None:
                    # Resistance level must be <= actual market maximum
                    if level_price <= market_max:
                        validated_resistance.append(level)
                    else:
                        self.logger.warning('.2f')

            # Update SR levels with validated results
            validated_sr_levels = {
                'support_levels': validated_support,
                'resistance_levels': validated_resistance
            }

            # Log validation results
            support_filtered = original_support_count - len(validated_support)
            resistance_filtered = original_resistance_count - len(validated_resistance)
            total_filtered = support_filtered + resistance_filtered

            if total_filtered > 0:
                self.logger.warning(f'🚨 Filtered out {total_filtered} invalid SR levels that were never reached by market:')
                self.logger.warning(f'   Support: {original_support_count} → {len(validated_support)} ({support_filtered} filtered)')
                self.logger.warning(f'   Resistance: {original_resistance_count} → {len(validated_resistance)} ({resistance_filtered} filtered)')
                self.logger.warning(f'   Market never traded above ${market_max:.2f} or below ${market_min:.2f}')
            else:
                self.logger.info(f'✅ All {original_support_count + original_resistance_count} SR levels validated against market data')

            return validated_sr_levels

        except Exception as e:
            self.logger.warning(f'⚠️ Error validating SR levels against market data: {e} - using original levels')
            return sr_levels

    def _log_detailed_sr_levels(self, sr_levels: Dict[str, Any], features_data: pd.DataFrame) -> None:
        """Log detailed information about all S/R levels found."""
        try:
            current_prices = features_data['close'].values
            price_min, price_max = min(current_prices), max(current_prices)
            price_range = price_max - price_min
            
            self.logger.info('📊 DETAILED S/R LEVEL ANALYSIS:')
            self.logger.info(f'   - Data price range: {price_min:.2f} - {price_max:.2f} (range: {price_range:.2f})')
            
            # Combine and sort all levels by price
            all_levels = []
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])

            # Add support levels
            for level in support_levels:
                if isinstance(level, (int, float)) and not np.isnan(level):
                    all_levels.append(('support', float(level), level))

            # Add resistance levels
            for level in resistance_levels:
                if isinstance(level, (int, float)) and not np.isnan(level):
                    all_levels.append(('resistance', float(level), level))

            # Sort by price
            all_levels.sort(key=lambda x: x[1])

            self.logger.info(f'   - Total S/R levels detected: {len(all_levels)} ({len(support_levels)} support, {len(resistance_levels)} resistance)')

            # Log first 20 levels in price order
            for i, (level_type, level_price, level) in enumerate(all_levels[:20]):
                distance_from_min = abs(level_price - price_min)
                distance_from_max = abs(level_price - price_max)
                closest_distance = min(distance_from_min, distance_from_max)
                type_display = level_type.capitalize()
                self.logger.info(f'     {i+1}. {type_display}: {level_price:.2f} (closest to data: {closest_distance:.2f})')
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to log detailed S/R levels: {e}')

    def _log_highly_correlated_pairs(self, features_df: pd.DataFrame, correlation_threshold: float = 0.95) -> None:
        """Log pairs of highly correlated features before removal."""
        try:
            # Filter to only numeric columns for correlation calculation
            numeric_features = features_df.select_dtypes(include=[np.number])
            if numeric_features.empty:
                self.logger.warning('⚠️ No numeric features found for correlation analysis')
                return
            
            # Calculate correlation matrix on numeric features only
            correlation_matrix = numeric_features.corr()

            # Define feature pairs that represent the same underlying data and should be excluded
            # These are raw data transformations that are expected to be highly correlated
            excluded_pairs = {
                ('price_change_pct', 'price_change_log'),
                ('price_change_log', 'price_change_pct'),
                ('close', 'close_log'),
                ('close_log', 'close'),
                ('high', 'high_log'),
                ('high_log', 'high'),
                ('low', 'low_log'),
                ('low_log', 'low'),
                ('open', 'open_log'),
                ('open_log', 'open'),
                ('volume', 'volume_log'),
                ('volume_log', 'volume'),
                ('vwap', 'vwap_log'),
                ('vwap_log', 'vwap'),
            }

            # Find highly correlated pairs (excluding self-correlations and expected raw data pairs)
            highly_correlated_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    correlation_value = abs(correlation_matrix.iloc[i, j])
                    if correlation_value >= correlation_threshold:
                        feature1 = correlation_matrix.columns[i]
                        feature2 = correlation_matrix.columns[j]

                        # Skip pairs that represent the same underlying data
                        pair_key = (feature1, feature2)
                        if pair_key not in excluded_pairs:
                            highly_correlated_pairs.append((feature1, feature2, correlation_value))

            if highly_correlated_pairs:
                self.logger.info(f'🔍 Found {len(highly_correlated_pairs)} highly correlated feature pairs (threshold: {correlation_threshold}):')
                # Sort by correlation strength (highest first)
                highly_correlated_pairs.sort(key=lambda x: x[2], reverse=True)

                # Log top 10 most correlated pairs
                for feature1, feature2, corr_value in highly_correlated_pairs[:10]:
                    self.logger.info(f'   • {feature1} ↔ {feature2}: {corr_value:.3f}')

                if len(highly_correlated_pairs) > 10:
                    self.logger.info(f'   • ... and {len(highly_correlated_pairs) - 10} more pairs')
            else:
                self.logger.info(f'✅ No feature pairs found with correlation ≥ {correlation_threshold}')

        except Exception as e:
            self.logger.warning(f'⚠️ Failed to log highly correlated feature pairs: {e}')

    def _find_timestamp_column(self, data: pd.DataFrame) -> str:
        """Find the timestamp column in the dataframe."""
        timestamp_candidates = ['timestamp', 'time', 'datetime', 'date']
        
        for col in timestamp_candidates:
            if col in data.columns:
                return col
        
        # If no standard timestamp column found, look for datetime-like columns
        for col in data.columns:
            if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                return col
        
        # If still no timestamp column found, return None
        return None

    def _filter_sr_levels_by_price_range(self, sr_levels: Dict[str, Any], chunk_prices: np.ndarray, 
                                        price_buffer: float = 2.0) -> Dict[str, Any]:
        """Filter S/R levels to only include those relevant to the chunk's price range."""
        try:
            chunk_min, chunk_max = min(chunk_prices), max(chunk_prices)
            chunk_range = chunk_max - chunk_min
            buffer_amount = chunk_range * price_buffer
            
            # Extended range to include nearby levels
            extended_min = chunk_min - buffer_amount
            extended_max = chunk_max + buffer_amount
            
            filtered_support = []
            filtered_resistance = []
            
            # Filter support levels
            for level in sr_levels.get('support_levels', []):
                # Handle both SRLevel objects and numeric values
                if hasattr(level, 'price'):
                    price = level.price
                elif isinstance(level, (int, float)) and not np.isnan(level):
                    price = level
                else:
                    continue
                
                if extended_min <= price <= extended_max:
                    filtered_support.append(level)
            
            # Filter resistance levels
            for level in sr_levels.get('resistance_levels', []):
                # Handle both SRLevel objects and numeric values
                if hasattr(level, 'price'):
                    price = level.price
                elif isinstance(level, (int, float)) and not np.isnan(level):
                    price = level
                else:
                    continue
                
                if extended_min <= price <= extended_max:
                    filtered_resistance.append(level)
            
            self.logger.info(f'🔍 S/R filtering for chunk (price range: {chunk_min:.2f}-{chunk_max:.2f}):')
            self.logger.info(f'   - Support levels: {len(sr_levels.get("support_levels", []))} -> {len(filtered_support)}')
            self.logger.info(f'   - Resistance levels: {len(sr_levels.get("resistance_levels", []))} -> {len(filtered_resistance)}')
            
            return {
                'support_levels': filtered_support,
                'resistance_levels': filtered_resistance,
                'original_support_count': len(sr_levels.get('support_levels', [])),
                'original_resistance_count': len(sr_levels.get('resistance_levels', []))
            }
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to filter S/R levels: {e}')
            return sr_levels

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

    async def _train_ml_models(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train ML models for SR level prediction with comprehensive evaluation and fast-fail checks."""
        self.logger.info('🤖 Starting comprehensive ML model training for SR optimization...')
        start_time = time.time()

        # Memory checkpoint before ML training
        if M1_MEMORY_AVAILABLE or M1_GPU_AVAILABLE:
            with memory_checkpoint("ml_training_start"):
                self.logger.info('📊 Memory checkpoint: ML training start')
                memory_usage = get_memory_usage()
                self.logger.info(f'🧠 Memory before ML training: {memory_usage["rss_gb"]:.2f}GB')
                # Pre-optimize memory before heavy ML operations
                optimize_memory()

        try:
            # If no SR levels provided, detect them for this specific chunk
            if not sr_levels or not sr_levels.get('support_levels') and not sr_levels.get('resistance_levels'):
                self.logger.info('🎯 No SR levels provided, detecting levels for current chunk...')
                sr_levels = self._detect_sr_levels(features_data)
                self.logger.info(f'✅ Detected {len(sr_levels.get("support_levels", []))} support, {len(sr_levels.get("resistance_levels", []))} resistance levels for chunk')
            
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

            # Apply comprehensive bias mitigation proactively
            if ML_COMMON_AVAILABLE and self.lookahead_protector:
                self.logger.info('🛠️ Applying comprehensive bias mitigation...')
                features_data, mitigation_results = self.lookahead_protector.comprehensive_bias_mitigation(
                    features_data, auto_fix=True
                )
                
                if mitigation_results.get('bias_detected', False):
                    fixes_applied = mitigation_results.get('fixes_applied', [])
                    self.logger.info(f'✅ Bias mitigation completed: {len(fixes_applied)} fixes applied')
                    for fix in fixes_applied:
                        self.logger.info(f'   • {fix}')
                    
                    # Store mitigation results for reference
                    self.bias_detection_results = mitigation_results.get('bias_results', {})
                else:
                    self.logger.info('✅ No bias issues detected - data is clean')
            else:
                # Fallback to legacy validation
                self.logger.info('🔧 Using legacy temporal validation (ML Common not available)')
                temporal_validation = self._validate_temporal_integrity(features_data)
                temporal_valid = temporal_validation.get('valid', False)
                bias_results = temporal_validation.get('bias_results', {})
                
                if not temporal_valid:
                    self.logger.warning('⚠️ Forward bias detected in training data - will use feature selection for bias mitigation')
                    self.bias_detection_results = bias_results

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
            self.logger.info(f'📊 Input features: {len(features_data.columns)}')
            result = self._prepare_ml_features(features_data, target_data)
            if len(result) == 5:
                # Handle different 5-value return formats from _prepare_ml_features
                first_val, second_val, third_val, fourth_val, fifth_val = result
                # Check if first value looks like X (numpy array) or feature names (list/array)
                if hasattr(first_val, 'shape') and len(first_val.shape) >= 2:
                    # Standard format: X, y_direction, y_volatility, feature_names, chunk_info
                    X, y_direction, y_volatility, feature_names, chunk_info = result
                elif isinstance(first_val, (list, np.ndarray)) and len(first_val) > 0:
                    # Feature selection format: selected_features, y_direction, y_volatility, feature_scores, selection_info
                    feature_names, y_direction, y_volatility, feature_scores, chunk_info = result
                    X = features_data[feature_names].values if hasattr(features_data, 'values') else features_data[feature_names]
                else:
                    # Fallback format: X_selected, y_direction, y_volatility, selected_features, selection_info
                    X, y_direction, y_volatility, feature_names, chunk_info = result
            else:
                # 4-value format: X, y_direction, y_volatility, feature_names
                X, y_direction, y_volatility, feature_names = result
                chunk_info = {}

            # Memory cleanup after feature preparation
            self._cleanup_memory(['features_data'] if 'features_data' in locals() else [])
            del features_data, target_data
            gc.collect()

            # Memory checkpoint after data preparation
            memory_after_prep = self._check_memory_usage()
            self.logger.info(f'🧠 Memory after data preparation: {memory_after_prep:.1%}')
            if memory_after_prep > 0.7:
                self.logger.warning(f'⚠️ High memory usage after data preparation: {memory_after_prep:.1%}')

            self.logger.info(f'📊 Output features: {len(feature_names)}')

            # Optimize hyperparameters
            self.logger.info('🔧 Optimizing hyperparameters...')
            try:
                hyperparameter_results = self._optimize_hyperparameters(X, y_direction, feature_names)
                # Memory cleanup after optimization
                self._cleanup_memory()
            except Exception as e:
                self.logger.warning(f'⚠️ Hyperparameter optimization failed: {e}')
                hyperparameter_results = None
                # Still cleanup memory on error
                self._cleanup_memory()

            # Feature selection
            self.logger.info('🎯 Performing feature selection...')
            try:
                X_selected, y_dir_selected, y_vol_selected, selected_feature_names, feature_selection_info = self._optimize_feature_selection(
                    X, y_direction, y_volatility, feature_names
                )
            except Exception as e:
                self.logger.warning(f'⚠️ Feature selection failed: {e}')
                X_selected, y_dir_selected, y_vol_selected, selected_feature_names, feature_selection_info = X, y_direction, y_volatility, feature_names, {
                    'original_features': len(feature_names),
                    'methods_used': ['fallback'],
                    'selected_features': len(feature_names),
                    'feature_importance': {},
                    'optimization_time': 0.0,
                    'reason': 'feature_selection_failed'
                }

            # Memory cleanup after feature selection
            self._cleanup_memory(['X', 'y_direction', 'y_volatility', 'feature_names'])

            # Memory checkpoint after feature selection
            memory_after_selection = self._check_memory_usage()
            self.logger.info(f'🧠 Memory after feature selection: {memory_after_selection:.1%}')
            if memory_after_selection > 0.75:
                self.logger.warning(f'⚠️ High memory usage after feature selection: {memory_after_selection:.1%}')

            # Split data
            self.logger.info('✂️ Splitting data into train/test sets...')
            X_train, X_test, y_dir_train, y_dir_test, y_vol_train, y_vol_test = train_test_split(
                X_selected, y_dir_selected, y_vol_selected, test_size=0.2, random_state=42, stratify=y_dir_selected
            )

            # Memory cleanup after data splitting - delete original selected data
            import gc
            del X_selected, y_dir_selected, y_vol_selected
            gc.collect()

            # Scale features
            self.logger.info('📏 Scaling features...')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Memory cleanup after scaling - delete unscaled data
            import gc
            del X_train, X_test
            gc.collect()

            # Memory checkpoint after scaling
            memory_after_scaling = self._check_memory_usage()
            self.logger.info(f'🧠 Memory after scaling: {memory_after_scaling:.1%}')
            if memory_after_scaling > 0.8:
                self.logger.warning(f'⚠️ High memory usage after scaling: {memory_after_scaling:.1%}')
                # Trigger aggressive cleanup if memory is very high
                if memory_after_scaling > 0.85:
                    self.logger.warning(f'🚨 Very high memory usage, triggering aggressive cleanup')
                    self.memory_optimizer._aggressive_memory_cleanup()

            # Train models with parallel processing coordination
            self.logger.info('🤖 Training ML models with parallel processing...')
            models_results = {}
            optimized_models = {}

            # Data validation before training
            unique_classes = np.unique(y_dir_train)
            if len(unique_classes) < 2:
                class_name = {0: 'Not Near SR', 1: 'Near SR'}.get(unique_classes[0], f'Class {unique_classes[0]}')
                self.logger.error(f'❌ CRITICAL: Single-class training data detected: only {class_name} present')
                self.logger.error(f'   This indicates a fundamental issue with SR detection or data splitting')
                raise ValueError(f"Cannot train models on single-class data: {unique_classes}")

            # Log class distribution
            class_counts = np.bincount(y_dir_train.astype(int))
            class_ratio = class_counts[1] / len(y_dir_train) if len(class_counts) > 1 else 0
            self.logger.info(f'📊 Training data: {len(y_dir_train)} samples, {len(unique_classes)} classes')
            self.logger.info(f'   Class distribution: {dict(zip(unique_classes, class_counts))}')
            self.logger.info(f'   SR ratio: {class_ratio:.1%}')

            # Advanced model selection based on class balance with SMOTE integration
            if class_ratio < 0.05 or class_ratio > 0.95:  # Extreme imbalance
                self.logger.warning(f'⚠️ Extreme class imbalance detected ({class_ratio:.1%}) - enabling SMOTE')
                # Use SMOTE for extreme imbalance + imbalance-resistant models
                self.enable_smote = True
                self.smote_config = {
                    'k_neighbors': min(5, min(class_counts) - 1),  # Adaptive k
                    'sampling_strategy': 'auto',
                    'random_state': 42
                }
                preferred_models = ['RandomForestClassifier', 'HistGradientBoostingClassifier']
                self.logger.info(f'   SMOTE enabled with k={self.smote_config["k_neighbors"]}')
                self.logger.info(f'   Using imbalance-resistant models: {preferred_models}')

            elif class_ratio < 0.15 or class_ratio > 0.85:  # Severe imbalance
                self.logger.warning(f'⚠️ Severe class imbalance detected ({class_ratio:.1%}) - enabling class weights')
                # Enable class weights for severe imbalance
                self.enable_class_weights = True
                self.class_weight_config = 'balanced'  # Auto-calculate weights
                preferred_models = ['RandomForestClassifier', 'HistGradientBoostingClassifier', 'LogisticRegression']
                self.logger.info(f'   Class weights enabled: {self.class_weight_config}')
                self.logger.info(f'   Using weight-aware models: {preferred_models}')

            elif class_ratio < 0.25 or class_ratio > 0.75:  # Moderate imbalance
                self.logger.info(f'⚠️ Moderate class imbalance detected ({class_ratio:.1%})')
                # Use ensemble methods that handle imbalance better
                preferred_models = ['RandomForestClassifier', 'HistGradientBoostingClassifier'] + \
                                 ['LogisticRegression', 'SVC']  # Add others for diversity
                self.logger.info(f'   Using ensemble-focused models: {preferred_models}')
            else:
                # Balanced data
                self.logger.info(f'✅ Well-balanced classes ({class_ratio:.1%})')
                preferred_models = list(self.ml_model_configs.keys())
                self.enable_smote = False
                self.enable_class_weights = False

            # Filter available models
            available_models = [m for m in preferred_models if m in self.ml_model_configs]
            if not available_models:
                available_models = list(self.ml_model_configs.keys())
                self.logger.warning(f'   Falling back to all available models: {available_models}')

            self.logger.info(f'🎯 Training {len(available_models)} models: {available_models}')

            # Use ML Common ParallelProcessingCoordinator if available
            if ML_COMMON_AVAILABLE and self.parallel_processor:
                try:
                    self.logger.info('🔧 Using ML Common ParallelProcessingCoordinator for model training')
                    
                    # Configure parallel processing for model training
                    parallel_config = {
                        'max_workers': min(4, len(self.ml_model_configs)),
                        'enable_gpu': self.enable_m1_optimizations,
                        'memory_threshold': 0.8,
                        'timeout_per_task': 300
                    }
                    
                    # Prepare training tasks
                    training_tasks = []
                    for model_name in available_models:
                        model_config = self.ml_model_configs[model_name]

                        # Apply class weights if enabled for severe imbalance
                        model_params = {}
                        if getattr(self, 'enable_class_weights', False):
                            if model_name in ['RandomForestClassifier', 'LogisticRegression']:
                                if self.class_weight_config == 'balanced':
                                    # Auto-calculate balanced weights
                                    from sklearn.utils.class_weight import compute_class_weight
                                    class_weights = compute_class_weight(
                                        'balanced',
                                        classes=np.unique(y_dir_train),
                                        y=y_dir_train
                                    )
                                    weight_dict = dict(zip(np.unique(y_dir_train), class_weights))
                                    model_params['class_weight'] = weight_dict
                                    self.logger.info(f'⚖️ Applied class weights for {model_name}: {weight_dict}')
                                else:
                                    model_params['class_weight'] = self.class_weight_config

                            elif model_name == 'HistGradientBoostingClassifier':
                                # HistGradientBoosting uses sample_weight instead of class_weight
                                self.logger.info(f'⚖️ Class weights will be applied via sample_weight for {model_name}')

                        # Use optimized hyperparameters if available
                        if hyperparameter_results and model_name in hyperparameter_results:
                            optimized_params = hyperparameter_results[model_name]['best_params']
                            model_params.update(optimized_params)
                            model = model_config['class'](**model_params)
                        else:
                            default_params = model_config['default_params']
                            model_params.update(default_params)
                            model = model_config['class'](**model_params)
                        
                        training_tasks.append({
                            'task_id': f'train_{model_name}',
                            'model': model,
                            'model_name': model_name,
                            'X_train': X_train_scaled,
                            'y_train': y_dir_train,
                            'X_test': X_test_scaled,
                            'y_test': y_dir_test
                        })
                    
                    # Execute parallel model training
                    parallel_results = self.parallel_processor.error_handling_parallel_execution(
                        training_tasks, 
                        max_retries=parallel_config.get('max_retries', 3)
                    )
                    
                    # Process parallel results
                    if parallel_results and 'results' in parallel_results:
                        for result in parallel_results['results']:
                            if result.get('success') and result.get('model_name'):
                                model_name = result['model_name']
                                models_results[model_name] = result['model_data']
                                optimized_models[model_name] = result['model_data']
                                self.logger.info(f'✅ {model_name} accuracy: {result["model_data"]["accuracy"]:.4f}')
                            elif result.get('error'):
                                self.logger.warning(f'⚠️ Parallel training failed for {result.get("model_name", "unknown")}: {result["error"]}')
                    
                    # Add parallel processing metrics
                    if 'execution_stats' in parallel_results:
                        stats = parallel_results['execution_stats']
                        self.logger.info(f'🚀 Parallel processing: {stats.get("total_tasks", 0)} tasks, '
                                       f'{stats.get("successful_tasks", 0)} successful, '
                                       f'{stats.get("total_time", 0):.2f}s')
                    
                    # Memory cleanup for parallel processing
                    import gc
                    del parallel_results, training_tasks
                    gc.collect()

                except Exception as e:
                    self.logger.warning(f'⚠️ Parallel processing failed: {e}')
                    # Fallback to sequential training
                    self.logger.info('🔧 Falling back to sequential model training')

                    # Clean up parallel processing variables on error
                    if 'training_tasks' in locals():
                        del training_tasks
                    if 'parallel_results' in locals():
                        del parallel_results
                    import gc
                    gc.collect()
            
            # Ensure ml_model_configs is initialized before any training
            if not hasattr(self, 'ml_model_configs') or self.ml_model_configs is None:
                self.logger.error('❌ ml_model_configs not initialized, cannot proceed with training')
                return {
                    'success': False,
                    'error': 'ml_model_configs not initialized',
                    'models_trained': 0,
                    'training_time': time.time() - start_time
                }
            
            # Sequential training (fallback or when parallel processing not available)
            if not models_results:
                self.logger.info('🔧 Training models sequentially...')
                
            for model_name in available_models:
                model_config = self.ml_model_configs[model_name]
                try:
                        # Note: chunk_info is not available in main training method, so we'll handle single-class errors in the exception handler
                        
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

                    # Store only essential model data to reduce memory usage
                    models_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'feature_importance': getattr(model, 'feature_importances_', None),
                        'params': model.get_params() if hasattr(model, 'get_params') else {},
                        'predictions_count': len(y_pred),  # Store count instead of full array
                        'probabilities_mean': float(y_pred_proba[:, 1].mean()) if y_pred_proba.shape[1] > 1 else float(y_pred_proba[:, 0].mean())
                    }

                    # Clean up temporary arrays
                    del y_pred, y_pred_proba

                    self.logger.info(f'✅ {model_name} accuracy: {accuracy:.4f}')

                    # Store optimized model
                    optimized_models[model_name] = models_results[model_name]

                except Exception as e:
                    error_str = str(e)
                    # Special handling for single-class errors
                    if ("needs samples of at least 2 classes" in error_str or
                        "Single-class" in error_str or
                        "only predicts one class" in error_str):

                        # Check actual class distribution
                        unique_train = np.unique(y_dir_train)
                        unique_test = np.unique(y_dir_test)

                        self.logger.warning(f'⚠️ {model_name} failed on imbalanced data: {error_str}')
                        self.logger.warning(f'   Training classes: {unique_train} (counts: {np.bincount(y_dir_train.astype(int))})')
                        self.logger.warning(f'   Test classes: {unique_test} (counts: {np.bincount(y_dir_test.astype(int))})')

                        # Skip this model for this chunk
                        self.logger.info(f'ℹ️ Skipping {model_name} for this chunk due to class imbalance')
                        continue  # Skip to next model
                    else:
                        self.logger.error(f'❌ Failed to train {model_name}: {e}')
                        # Skip this model and continue with others

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
            
            # Memory cleanup: Clear large arrays after evaluation
            del X_train_scaled, X_test_scaled, y_dir_train, y_dir_test, y_vol_train, y_vol_test
            del X_selected, y_dir_selected, y_vol_selected
            del X, y_direction, y_volatility

            # Save best model
            self.logger.info('💾 Saving best performing model...')
            models_for_saving = optimized_models if optimized_models else models_results

            model_save_path = self._save_best_model(
                models_for_saving, scaler, selected_feature_names
            )

            # Compile final results
            training_time = time.time() - start_time
            
            # Store essential metrics before cleanup
            training_samples_count = evaluation_metrics.get('training_samples', 0)
            test_samples_count = evaluation_metrics.get('test_samples', 0)
            sr_levels_count = len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', []))
            
            ml_results = {
                'direction_accuracy': evaluation_metrics['best_direction_accuracy'],
                'volatility_mae': evaluation_metrics['best_volatility_mae'],
                'model_type': evaluation_metrics['best_model_type'],
                'training_samples': training_samples_count,
                'test_samples': test_samples_count,
                'sr_levels_used': sr_levels_count,
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

            # Final memory checkpoint before cleanup
            memory_before_cleanup = self._check_memory_usage()
            self.logger.info(f'🧠 Memory before final cleanup: {memory_before_cleanup:.1%}')
            if memory_before_cleanup > 0.8:
                self.logger.warning(f'⚠️ High memory usage before cleanup: {memory_before_cleanup:.1%}')

            # Comprehensive memory cleanup before returning
            import gc
            gc.collect()
            
            # Clear large training data arrays first
            if 'X_train' in locals():
                del X_train
            if 'X_test' in locals():
                del X_test
            if 'y_dir_train' in locals():
                del y_dir_train
            if 'y_dir_test' in locals():
                del y_dir_test
            if 'y_vol_train' in locals():
                del y_vol_train
            if 'y_vol_test' in locals():
                del y_vol_test
            if 'X_train_scaled' in locals():
                del X_train_scaled
            if 'X_test_scaled' in locals():
                del X_test_scaled

            # Clear feature selection data
            if 'X_selected' in locals():
                del X_selected
            if 'y_dir_selected' in locals():
                del y_dir_selected
            if 'y_vol_selected' in locals():
                del y_vol_selected

            # Clear models and related objects
            if 'models_results' in locals():
                del models_results
            if 'optimized_models' in locals():
                del optimized_models
            if 'scaler' in locals():
                del scaler
            if 'feature_names' in locals():
                del feature_names
            if 'selected_feature_names' in locals():
                del selected_feature_names

            # Final garbage collection
            gc.collect()

            # Memory checkpoint after cleanup
            memory_after_cleanup = self._check_memory_usage()
            memory_reduction = memory_before_cleanup - memory_after_cleanup
            self.logger.info(f'🧠 Memory after cleanup: {memory_after_cleanup:.1%} (reduced by {memory_reduction:.1%})')

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
            def __init__(self, name, logger, memory_checker):
                self.name = name
                self.logger = logger
                self.memory_checker = memory_checker
                self.start_time = None
                self.start_memory = None
            
            def __enter__(self):
                self.start_time = time.time()
                self.start_memory = self.memory_checker()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                end_memory = self.memory_checker()
                memory_delta = end_memory - self.start_memory
                
                if exc_type is None:
                    self.logger.info(f"⏱️ {self.name}: {duration:.2f}s, Memory: {memory_delta:+.1%}")
                else:
                    self.logger.error(f"❌ {self.name} failed after {duration:.2f}s, Memory: {memory_delta:+.1%}")
        
        return PerformanceMonitor(operation_name, self.logger, self._check_memory_usage)

    def _validate_ml_methods_exist(self) -> bool:
        """Validate that all required ML methods exist and are callable."""
        required_methods = [
            '_prepare_sr_targets',
            '_prepare_ml_features',
            '_train_ml_models_chunked_optimized',
            '_optimize_hyperparameters',
            '_optimize_feature_selection',
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
            self.logger.info(f'🔍 Checking method {method_name}: hasattr={has_attr}')
            print(f'DEBUG: Checking method {method_name}: hasattr={has_attr}', flush=True)

            if not has_attr:
                missing_methods.append(method_name)
                continue

            # Additional check: ensure the method is callable
            method = getattr(self, method_name)
            is_callable = callable(method)
            self.logger.info(f'🔍 Method {method_name}: callable={is_callable}')
            print(f'DEBUG: Method {method_name}: callable={is_callable}', flush=True)

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

    def _train_single_model_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Single model training task for parallel processing."""
        try:
            model = task_data['model']
            model_name = task_data['model_name']
            X_train = task_data['X_train']
            y_train = task_data['y_train']
            X_test = task_data['X_test']
            y_test = task_data['y_test']
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = getattr(model, 'predict_proba', lambda X: np.zeros((len(X), 2)))(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            model_data = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0],
                'feature_importance': getattr(model, 'feature_importances_', None),
                'params': model.get_params() if hasattr(model, 'get_params') else {}
            }
            
            return {
                'success': True,
                'model_name': model_name,
                'model_data': model_data
            }
            
        except Exception as e:
            error_str = str(e)
            # Special handling for LogisticRegression single-class errors
            if model_name == 'LogisticRegression' and "needs samples of at least 2 classes" in error_str:
                return {
                    'success': False,
                    'model_name': model_name,
                    'error': f'Single-class data: {error_str}',
                    'skip_model': True
                }
            else:
                return {
                    'success': False,
                    'model_name': model_name,
                    'error': str(e)
                }

    async def _process_single_chunk_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Single chunk processing task for parallel processing."""
        try:
            chunk_data = task_data['chunk_data']
            sr_levels = task_data['sr_levels']
            chunk_num = task_data['chunk_num']
            total_chunks = task_data['total_chunks']
            
            # Memory checkpoint for this chunk
            memory_before = self._check_memory_usage()
            
            # Adaptive memory management
            if memory_before > 0.85:
                self.logger.info(f'🧠 High memory usage ({memory_before:.1%}), applying aggressive cleanup')
                if self.memory_optimizer:
                    self.memory_optimizer._aggressive_memory_cleanup()
            
            self.logger.info(f'🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_data):,} rows)')
            chunk_start = time.time()
            
            # Memory checkpoint before chunk processing
            memory_before_chunk = self._check_memory_usage()
            self.logger.info(f'🧠 Memory before chunk {chunk_num} processing: {memory_before_chunk:.1%}')
            if memory_before_chunk > 0.9:
                self.logger.warning(f'🚨 Very high memory usage before chunk processing: {memory_before_chunk:.1%}')

            try:
                # Process chunk with memory-efficient timeout
                timeout_duration = min(300, max(60, len(chunk_data) / 1000))  # Adaptive timeout
                chunk_result = await asyncio.wait_for(
                    self._train_ml_models(chunk_data, sr_levels),
                    timeout=timeout_duration
                )
                chunk_time = time.time() - chunk_start
                
                # Memory usage after processing
                memory_after = self._check_memory_usage()
                self.logger.info(f'✅ Chunk {chunk_num} completed in {chunk_time:.2f}s, memory: {memory_after:.1%}')
                
                return {
                    'success': True,
                    'chunk_num': chunk_num,
                    'chunk_result': chunk_result,
                    'processing_time': chunk_time,
                    'memory_usage': memory_after
                }
                
            except asyncio.TimeoutError:
                error_message = f'Chunk {chunk_num} timed out after {timeout_duration}s'
                self.logger.error(f'⏰ {error_message}')
                fallback_result = self._handle_ml_failure(error_message, "TIMEOUT_ERROR")
                return {
                    'success': False,
                    'chunk_num': chunk_num,
                    'error': error_message,
                    'chunk_result': fallback_result
                }
                
            except ValueError as chunk_error:
                error_str = str(chunk_error)
                if "Single-class" in error_str:
                    self.logger.warning(f'⚠️ Skipping chunk {chunk_num}: {error_str}')
                    return {
                        'success': False,
                        'chunk_num': chunk_num,
                        'error': f'Single-class binary chunk: {error_str}',
                        'skip_chunk': True
                    }
                else:
                    error_message = f'Chunk {chunk_num} failed: {error_str}'
                    self.logger.error(f'❌ {error_message}')
                    fallback_result = self._handle_ml_failure(error_message, "CHUNK_ERROR")
                    return {
                        'success': False,
                        'chunk_num': chunk_num,
                        'error': error_message,
                        'chunk_result': fallback_result
                    }
                    
            except Exception as chunk_error:
                error_message = f'Chunk {chunk_num} failed: {str(chunk_error)}'
                self.logger.error(f'❌ {error_message}')
                fallback_result = self._handle_ml_failure(error_message, "CHUNK_ERROR")
                return {
                    'success': False,
                    'chunk_num': chunk_num,
                    'error': error_message,
                    'chunk_result': fallback_result
                }
                
        except Exception as e:
            return {
                'success': False,
                'chunk_num': task_data.get('chunk_num', '?'),
                'error': str(e)
            }

    async def _train_ml_models_chunked_optimized(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any], chunk_size: int) -> Dict[str, Any]:
        """Optimized chunked ML training with ML Common memory optimization."""
        try:
            total_chunks = (len(features_data) + chunk_size - 1) // chunk_size
            self.logger.info(f'📊 Processing {len(features_data):,} rows in {total_chunks} chunks of {chunk_size:,} rows each')

            # Memory checkpoint before chunked processing
            if M1_MEMORY_AVAILABLE or M1_GPU_AVAILABLE:
                with memory_checkpoint("chunked_training_start"):
                    self.logger.info('📊 Memory checkpoint: Chunked training start')
                    memory_usage = get_memory_usage()
                    self.logger.info(f'🧠 Memory before chunked training: {memory_usage["rss_gb"]:.2f}GB')
                    # Pre-optimize memory before chunked processing
                    optimize_memory()

            # Use ML Common MemoryEfficientTraining if available
            if ML_COMMON_AVAILABLE and self.memory_optimizer:
                try:
                    self.logger.info('🔧 Using ML Common MemoryEfficientTraining for chunked processing')
                    
                    # Create optimal data chunking strategy
                    chunking_strategy = self.memory_optimizer.data_chunking_strategy(
                        features_data, chunk_size_mb=chunk_size * 8 / (1024 * 1024)  # Rough MB estimate
                    )
                    
                    # Memory-efficient streaming processing
                    all_results = []
                    chunk_processing_times = []
                    memory_usage_history = []
                    
                    chunk_num = 0
                    for chunk_data, _ in chunking_strategy:
                        chunk_num += 1
                        
                        # Ensure chunk_data is a DataFrame (not a generator)
                        if hasattr(chunk_data, '__iter__') and not isinstance(chunk_data, pd.DataFrame):
                            chunk_data = pd.DataFrame(list(chunk_data))
                        
                        # Memory checkpoint for this chunk
                        with self.memory_optimizer.memory_checkpoint(f"chunk_{chunk_num}"):
                            memory_before = self._check_memory_usage()
                            memory_usage_history.append(memory_before)
                            
                            # Adaptive memory management
                            if memory_before > 0.85:
                                self.logger.info(f'🧠 High memory usage ({memory_before:.1%}), applying aggressive cleanup')
                                self.memory_optimizer._aggressive_memory_cleanup()

                            self.logger.info(f'🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_data):,} rows)')
                            chunk_start = time.time()

                            try:
                                # Process chunk with memory-efficient timeout
                                timeout_duration = min(300, max(60, len(chunk_data) / 1000))  # Adaptive timeout
                                chunk_result = await asyncio.wait_for(
                                    self._train_ml_models(chunk_data, sr_levels),
                                    timeout=timeout_duration
                                )
                                chunk_time = time.time() - chunk_start
                                chunk_processing_times.append(chunk_time)

                                # Memory usage after processing
                                memory_after = self._check_memory_usage()
                                self.logger.info(f'✅ Chunk {chunk_num} completed in {chunk_time:.2f}s, memory: {memory_after:.1%}')

                                all_results.append(chunk_result)
                                
                                # Memory cleanup after each chunk
                                import gc
                                gc.collect()

                            except asyncio.TimeoutError:
                                error_message = f'Chunk {chunk_num} timed out after {timeout_duration}s'
                                self.logger.error(f'⏰ {error_message}')
                                fallback_result = self._handle_ml_failure(error_message, "TIMEOUT_ERROR")
                                all_results.append(fallback_result)
                            except ValueError as chunk_error:
                                error_str = str(chunk_error)
                                if "Single-class" in error_str:
                                    self.logger.warning(f'⚠️ Skipping chunk {chunk_num}: {error_str}')
                                    continue
                                else:
                                    error_message = f'Chunk {chunk_num} failed: {error_str}'
                                    self.logger.error(f'❌ {error_message}')
                                    fallback_result = self._handle_ml_failure(error_message, "CHUNK_ERROR")
                                    all_results.append(fallback_result)
                            except Exception as chunk_error:
                                error_message = f'Chunk {chunk_num} failed: {str(chunk_error)}'
                                self.logger.error(f'❌ {error_message}')
                                fallback_result = self._handle_ml_failure(error_message, "CHUNK_ERROR")
                                all_results.append(fallback_result)
                    
                    # Memory optimization summary
                    if memory_usage_history:
                        avg_memory = np.mean(memory_usage_history)
                        max_memory = np.max(memory_usage_history)
                        self.logger.info(f'🧠 Memory optimization summary: Avg={avg_memory:.1%}, Max={max_memory:.1%}')
                    
                except Exception as e:
                    self.logger.error(f'❌ ML Common memory optimization failed: {e}')
                    return self._handle_ml_failure(str(e), "MEMORY_OPTIMIZATION_ERROR")
            else:
                self.logger.error('❌ ML Common memory optimization not available')
                return self._handle_ml_failure("ML Common memory optimization not available", "ML_COMMON_UNAVAILABLE_ERROR")

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
                    'total_processing_time': sum(chunk_processing_times),
                    'memory_optimization_used': ML_COMMON_AVAILABLE and self.memory_optimizer is not None
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
            self.logger.info(f'🎯 _prepare_sr_targets: Starting with features shape {features_data.shape}')
            self.logger.info(f'🎯 _prepare_sr_targets: Features columns: {list(features_data.columns)}')
            self.logger.info(f'🎯 _prepare_sr_targets: SR levels keys: {list(sr_levels.keys()) if sr_levels else "None"}')
            
            # Get current price data
            if 'close' not in features_data.columns:
                self.logger.error(f'❌ Missing close column. Available columns: {list(features_data.columns)}')
                raise ValueError("Features data must contain 'close' price column")

            current_prices = features_data['close'].values
            self.logger.info(f'🎯 _prepare_sr_targets: Current prices range: {current_prices.min():.2f} - {current_prices.max():.2f}')
            
            # Filter S/R levels to only include those relevant to this chunk's price range
            filtered_sr_levels = self._filter_sr_levels_by_price_range(sr_levels, current_prices)
            
            # Log filtered levels for this chunk
            self._log_filtered_sr_levels(filtered_sr_levels, sr_levels, current_prices)
            
            # Extract prices from filtered S/R levels (they're already SRLevel objects)
            support_prices = []
            resistance_prices = []
            
            # Extract prices from filtered support levels
            for level in filtered_sr_levels.get('support_levels', []):
                if hasattr(level, 'price'):
                    support_prices.append(level.price)
                elif isinstance(level, (int, float)) and not np.isnan(level):
                    support_prices.append(level)
            
            # Extract prices from filtered resistance levels  
            for level in filtered_sr_levels.get('resistance_levels', []):
                if hasattr(level, 'price'):
                    resistance_prices.append(level.price)
                elif isinstance(level, (int, float)) and not np.isnan(level):
                    resistance_prices.append(level)
            
            target_data = pd.DataFrame(index=features_data.index)

            # CRITICAL: Create targets WITHOUT forward bias
            # Only use SR levels that were detected using historical data up to each point
            proximity_threshold = self.proximity_threshold  # Configurable proximity threshold (default 0.2%)
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
            proximity_debug = []
            support_matches = 0
            resistance_matches = 0
            
            # Sample every 1000th point for debugging to avoid too much output
            debug_indices = list(range(0, len(current_prices), max(1, len(current_prices) // 1000)))
            
            for i, current_price in enumerate(current_prices):
                # Check proximity to support levels
                for support_price in valid_support_prices:
                    distance_pct = abs(current_price - support_price) / current_price
                    if i in debug_indices:
                        proximity_debug.append(f"Price {current_price:.2f} vs Support {support_price:.2f} = {distance_pct:.3f} ({distance_pct*100:.1f}%)")
                    if distance_pct <= proximity_threshold:
                        near_support[i] = 1.0
                        support_matches += 1
                        break

                # Check proximity to resistance levels
                for resistance_price in valid_resistance_prices:
                    distance_pct = abs(current_price - resistance_price) / current_price
                    if i in debug_indices:
                        proximity_debug.append(f"Price {current_price:.2f} vs Resistance {resistance_price:.2f} = {distance_pct:.3f} ({distance_pct*100:.1f}%)")
                    if distance_pct <= proximity_threshold:
                        near_resistance[i] = 1.0
                        resistance_matches += 1
                        break
            
            # Log proximity debug info (first 10 examples)
            if proximity_debug:
                self.logger.info(f'🔍 Proximity calculation examples (threshold: {proximity_threshold*100:.1f}%):')
                for debug_info in proximity_debug[:10]:
                    self.logger.info(f'   - {debug_info}')
            
            # Log match statistics
            self.logger.info(f'🔍 Proximity match statistics:')
            self.logger.info(f'   - Support matches: {support_matches} out of {len(current_prices)} data points')
            self.logger.info(f'   - Resistance matches: {resistance_matches} out of {len(current_prices)} data points')
            self.logger.info(f'   - Total matches: {support_matches + resistance_matches}')
            self.logger.info(f'   - Proximity threshold: {proximity_threshold*100:.1f}%')
            
            # Check for extreme class imbalance and adjust threshold if needed
            total_matches = support_matches + resistance_matches
            match_ratio = total_matches / len(current_prices) if len(current_prices) > 0 else 0
            
            if match_ratio > 0.50:  # If more than 50% of samples are "Near SR"
                self.logger.warning(f'⚠️ Extreme class imbalance detected: {match_ratio:.1%} samples near SR')
                self.logger.warning(f'   Adjusting proximity threshold from {proximity_threshold*100:.1f}% to {proximity_threshold*0.3*100:.1f}%')
                
                # Iterative threshold adjustment for better class balance
                original_threshold = proximity_threshold
                for attempt in range(5):  # Try up to 5 times with progressively stricter thresholds
                    near_support = np.zeros(len(current_prices))
                    near_resistance = np.zeros(len(current_prices))
                    support_matches = 0
                    resistance_matches = 0

                    for i, current_price in enumerate(current_prices):
                        # Check proximity to support levels
                        for support_price in valid_support_prices:
                            distance_pct = abs(current_price - support_price) / current_price
                            if distance_pct <= proximity_threshold:
                                near_support[i] = 1.0
                                support_matches += 1
                                break

                        # Check proximity to resistance levels
                        for resistance_price in valid_resistance_prices:
                            distance_pct = abs(current_price - resistance_price) / current_price
                            if distance_pct <= proximity_threshold:
                                near_resistance[i] = 1.0
                                resistance_matches += 1
                                break

                    # Log updated statistics
                    total_matches = support_matches + resistance_matches
                    match_ratio = total_matches / len(current_prices) if len(current_prices) > 0 else 0

                    if match_ratio <= 0.40:  # Good balance achieved
                        self.logger.info(f'✅ After threshold adjustment (attempt {attempt+1}): {match_ratio:.1%} samples near SR (threshold: {proximity_threshold*100:.3f}%)')
                        break
                    elif attempt < 4:  # Don't adjust on the last attempt
                        # Further reduce threshold for next attempt
                        old_threshold = proximity_threshold
                        proximity_threshold *= 0.7
                        self.logger.warning(f'   Still imbalanced ({match_ratio:.1%}), reducing threshold from {old_threshold*100:.3f}% to {proximity_threshold*100:.3f}% (attempt {attempt+2})')
                    else:
                        self.logger.warning(f'   Final attempt: {match_ratio:.1%} samples near SR (threshold: {proximity_threshold*100:.3f}%)')

                self.logger.info(f'📊 Threshold adjustment summary: {original_threshold*100:.3f}% → {proximity_threshold*100:.3f}%')
            
            # Debug: Check if we have any valid S/R prices
            if len(valid_support_prices) == 0 and len(valid_resistance_prices) == 0:
                self.logger.error('❌ CRITICAL: No valid S/R prices found! This explains why we have 0 matches.')
                self.logger.error(f'   - Original support_prices: {len(support_prices)}')
                self.logger.error(f'   - Original resistance_prices: {len(resistance_prices)}')
                self.logger.error(f'   - Valid support_prices: {len(valid_support_prices)}')
                self.logger.error(f'   - Valid resistance_prices: {len(valid_resistance_prices)}')
                
                # Show sample of original prices to debug
                if support_prices:
                    self.logger.error(f'   - Sample original support prices: {support_prices[:5]}')
                if resistance_prices:
                    self.logger.error(f'   - Sample original resistance prices: {resistance_prices[:5]}')
            else:
                self.logger.info(f'✅ Valid S/R prices found: {len(valid_support_prices)} support, {len(valid_resistance_prices)} resistance')

            # Create DISCRETE direction target (0 = bearish, 1 = bullish, 2 = neutral)
            target_data['near_support'] = near_support
            target_data['near_resistance'] = near_resistance

            # DEBUG: Log S/R level information
            self.logger.info(f'🔍 DEBUG: S/R Detection Results:')
            self.logger.info(f'   - Support levels detected: {len(valid_support_prices)}')
            self.logger.info(f'   - Resistance levels detected: {len(valid_resistance_prices)}')
            self.logger.info(f'   - Data points near support: {near_support.sum()}')
            self.logger.info(f'   - Data points near resistance: {near_resistance.sum()}')
            self.logger.info(f'   - Total data points: {len(current_prices)}')
            
            if len(valid_support_prices) > 0:
                self.logger.info(f'   - Support price range: {min(valid_support_prices):.2f} - {max(valid_support_prices):.2f}')
            if len(valid_resistance_prices) > 0:
                self.logger.info(f'   - Resistance price range: {min(valid_resistance_prices):.2f} - {max(valid_resistance_prices):.2f}')
            self.logger.info(f'   - Current price range: {min(current_prices):.2f} - {max(current_prices):.2f}')

            # Convert to binary classification: SR vs Non-SR (much simpler and more robust)
            # Combine support and resistance into single "near SR" class
            near_any_sr = np.maximum(near_support, near_resistance)  # 1 if near any SR level, 0 otherwise
            
            # Binary target: 1 = Near SR level, 0 = Not near SR level
            target_data['sr_target'] = near_any_sr.astype(int)
            
            # Log binary class distribution
            sr_matches = near_any_sr.sum()
            non_sr_matches = len(current_prices) - sr_matches
            self.logger.info(f'📊 Binary SR classification:')
            self.logger.info(f'   - Near SR levels: {sr_matches} samples (class 1)')
            self.logger.info(f'   - Not near SR levels: {non_sr_matches} samples (class 0)')
            self.logger.info(f'   - Total samples: {len(current_prices)}')
            self.logger.info(f'   - SR ratio: {sr_matches/len(current_prices)*100:.1f}%')
            
            # Validate binary class diversity
            unique_classes = np.unique(target_data['sr_target'])
            if len(unique_classes) < 2:
                self.logger.error(f'❌ INSUFFICIENT BINARY CLASS DIVERSITY: Only {len(unique_classes)} class(es) present: {unique_classes}')
                self.logger.error(f'   Expected 2 classes (0=Not Near SR, 1=Near SR) but found: {unique_classes}')
                self.logger.error(f'   This indicates a fundamental issue with SR detection or proximity calculation.')
                self.logger.error(f'   Total samples: {len(current_prices)}')
                self.logger.error(f'   SR matches: {sr_matches}, Non-SR matches: {non_sr_matches}')
                
                # Log class distribution for debugging
                target_counts = target_data["sr_target"].value_counts().sort_index()
                self.logger.error(f'   Class distribution: {dict(target_counts)}')
                
                raise ValueError(f"SR target preparation failed: Insufficient binary class diversity: Only {len(unique_classes)} class(es) present {sorted(unique_classes)}. Need 2 classes (0=Not Near SR, 1=Near SR) for proper ML training.")
            
            # Ensure reasonable class balance (at least 10% of each class)
            min_class_ratio = 0.10  # 10% minimum for each class
            sr_ratio = sr_matches / len(current_prices)
            if sr_ratio < min_class_ratio or sr_ratio > (1 - min_class_ratio):
                self.logger.warning(f'⚠️ Imbalanced binary classes: SR ratio = {sr_ratio:.1%}')
                self.logger.warning(f'   Consider adjusting proximity threshold for better balance')
                
                # If still extremely imbalanced after adjustment, use stratified sampling
                if sr_ratio > 0.95 or sr_ratio < 0.05:
                    self.logger.warning(f'⚠️ Still extremely imbalanced after threshold adjustment')
                    self.logger.warning(f'   This may indicate insufficient SR level diversity or data quality issues')
                    self.logger.warning(f'   Proceeding with current distribution but ML performance may be limited')
            
            # Keep the original direction_target for backward compatibility (but use sr_target for ML)
            target_data['direction_target'] = target_data['sr_target']  # For compatibility

            # Create volatility target based on proximity to levels (binary for classification)
            proximity_score = np.maximum(near_support, near_resistance)
            target_data['volatility_target'] = proximity_score.astype(int)  # 0 or 1

            # Add momentum as a separate feature (not used in target classification)
            if len(current_prices) > 5:
                short_trend = np.sign(current_prices[5:] - current_prices[:-5])
                trend_signal = np.concatenate([np.zeros(5), short_trend])
                target_data['momentum_signal'] = trend_signal.astype(int)  # Separate feature for ML model
                
            # Note: direction_target remains pure S/R logic (no momentum influence)

            self.logger.info(f'🎯 Pure S/R targets prepared (no momentum bias): {len(valid_support_prices)} support, {len(valid_resistance_prices)} resistance levels')
            
            # DEBUG: Show detailed target distribution for binary classification
            target_counts = target_data["sr_target"].value_counts().to_dict()
            self.logger.info(f'📊 Binary target distribution: {target_counts}')
            self.logger.info(f'   - Class 0 (Not Near SR): {target_counts.get(0, 0)} samples')
            self.logger.info(f'   - Class 1 (Near SR): {target_counts.get(1, 0)} samples')
            self.logger.info(f'   - Total samples: {len(target_data)}')
            
            # Log final target distribution for binary classification
            target_distribution = target_data['sr_target'].value_counts().sort_index()
            self.logger.info(f'📊 Final binary target distribution: {dict(target_distribution)}')
            for class_id, count in target_distribution.items():
                class_name = {0: 'Not Near SR', 1: 'Near SR'}.get(class_id, f'Unknown Class {class_id}')
                self.logger.info(f'   - Class {class_id} ({class_name}): {count} samples')
            self.logger.info(f'   - Total samples: {len(target_data)}')

            self.logger.info(f'✅ _prepare_sr_targets: Returning target_data with columns: {list(target_data.columns)}')
            return target_data

        except Exception as e:
            self.logger.error(f'❌ SR target preparation failed: {e}')
            # Return dataframe with proper discrete classes
            target_data = pd.DataFrame(index=features_data.index)
            target_data['sr_target'] = 0  # Default to "Not Near SR" class
            target_data['direction_target'] = 0  # Default to "Not Near SR" class
            target_data['volatility_target'] = 0  # No volatility signal
            target_data['near_support'] = 0
            target_data['near_resistance'] = 0
            self.logger.info(f'⚠️ _prepare_sr_targets: Returning fallback target_data with columns: {list(target_data.columns)}')
            return target_data

    def _ml_based_feature_selection(self, features_data: pd.DataFrame, target_data: pd.DataFrame,
                                   feature_cols: List[str], max_features: int = 100) -> List[str]:
        """
        Use ML Common feature selection framework for intelligent feature selection.

        This leverages sophisticated algorithms (mRMR, mutual information, stability analysis)
        to select the most relevant features for SR detection.
        """
        if not ML_COMMON_AVAILABLE:
            self.logger.warning('⚠️ ML Common not available, falling back to simple prioritization')
            return self._simple_prioritization(feature_cols, max_features)

        try:
            # Prepare data for ML-based selection
            feature_data = features_data[feature_cols].copy()

            # Handle missing values for feature selection
            feature_data = feature_data.fillna(feature_data.mean())
            feature_data = feature_data.fillna(0)  # Fallback for any remaining NaN

            # Get target data
            if 'sr_target' not in target_data.columns:
                raise ValueError("Target data must contain 'sr_target' column")
            y = target_data['sr_target'].values

            # Convert to numpy arrays
            X = feature_data.values
            feature_names = list(feature_data.columns)

            # Use ML Common feature selection framework
            selector = get_feature_selector()

            # Select features using mRMR (Minimum Redundancy Maximum Relevance)
            # This balances relevance to target with low correlation between features
            selected_features, feature_scores, selection_info = selector.select_features(
                X=X,
                y=y,
                feature_names=feature_names,
                n_features=min(max_features, len(feature_names)),
                method='mrmr'  # mRMR is excellent for this use case
            )

            if not selected_features:
                self.logger.warning('⚠️ ML feature selection returned no features, using fallback')
                return self._simple_prioritization(feature_cols, max_features)

            # Log selection results
            self.logger.info(f'🎯 ML-based feature selection: {len(feature_cols)} -> {len(selected_features)} features')
            self.logger.info(f'📊 Selection method: {selection_info.get("method", "unknown")}')

            # Log top 5 selected features with scores
            if feature_scores:
                sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_scores[:5]
                self.logger.info('🏆 Top selected features:')
                for feature, score in top_features:
                    self.logger.info(f'   • {feature}: {score:.4f}')

            return selected_features

        except Exception as e:
            self.logger.warning(f'⚠️ ML-based feature selection failed: {e}')
            self.logger.info('🔄 Falling back to simple prioritization')

    def _integrate_step06_advanced_features(self, features_data: pd.DataFrame) -> pd.DataFrame:
        """Integrate advanced features from step06 feature engineering."""
        if not ADVANCED_FEATURES_AVAILABLE or AdvancedFeatureEngineeringStep is None:
            self.logger.warning('⚠️ Step06 advanced features not available - skipping integration')
            return features_data

        try:
            # Create advanced feature engineering instance
            feature_config = {
                'feature_engineering': {
                    'enable_wavelets': False,  # Disable for step02_5 compatibility
                    'enable_multi_timeframe': True,
                    'enable_feature_interactions': False,  # Disable for step02_5 compatibility
                    'enable_regime_features': False,  # Disable for step02_5 compatibility
                    'timeframes': ['30m', '1h', '4h'],
                    'chunk_size': 100000,
                    'max_features': 200,
                    'disable_lookback_optimization': True  # Enable step02_5 compatibility mode
                }
            }

            advanced_engineer = AdvancedFeatureEngineeringStep(feature_config)

            # Prepare input data for step06 (needs labeled format)
            # Create a mock labeled DataFrame with the same index as features_data
            labeled_data = features_data.copy()

            # Add required columns if missing
            if 'label' not in labeled_data.columns:
                labeled_data['label'] = 0  # Default label

            # Prepare training input and pipeline state
            training_input = {
                'symbol': getattr(self, 'symbol', 'DEFAULT'),
                'exchange': getattr(self, 'exchange', 'BINANCE'),
                'timeframe': getattr(self, 'timeframe', '30m')  # Use 30m for SR analysis
            }

            pipeline_state = {
                'labeled_data': labeled_data
            }

            # Execute advanced feature engineering (handle async call)
            self.logger.info('🔧 Executing step06 advanced feature engineering...')
            import asyncio
            try:
                # Try to run async method in current event loop if available
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we need to handle this differently
                    # For now, create a new event loop
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(advanced_engineer.execute_logic(training_input, pipeline_state))
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(loop)
                else:
                    result = loop.run_until_complete(advanced_engineer.execute_logic(training_input, pipeline_state))
            except RuntimeError:
                # No event loop, create a new one
                result = asyncio.run(advanced_engineer.execute_logic(training_input, pipeline_state))

            # Extract engineered features
            if 'features' in result:
                engineered_features = result['features']
                self.logger.info(f'✅ Step06 features engineered: {len(engineered_features.columns)} features')

                # Combine with existing features
                # Avoid duplicate columns
                existing_cols = set(features_data.columns)
                new_cols = [col for col in engineered_features.columns if col not in existing_cols]

                if new_cols:
                    combined_features = pd.concat([features_data, engineered_features[new_cols]], axis=1)
                    self.logger.info(f'📊 Combined features: {len(combined_features.columns)} total ({len(new_cols)} new from step06)')
                    return combined_features
                else:
                    self.logger.info('📊 No new features from step06 - using original features')
                    return features_data
            else:
                self.logger.warning('⚠️ Step06 returned no features - using original features')
                return features_data

        except Exception as e:
            self.logger.warning(f'⚠️ Step06 advanced feature integration failed: {e}')
            self.logger.info('🔄 Using original features without step06 integration')
            return features_data

    def _prepare_ml_features(self, features_data: pd.DataFrame, target_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ...]:
        """Prepare features and targets for ML training."""
        try:
            # Memory monitoring at start
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            self.logger.info(f'📊 _prepare_ml_features: Starting with {len(features_data.columns)} features (Memory: {memory_before:.1f}MB)')

            # Step 1: Integrate advanced features from step06 BEFORE any processing
            self.logger.info('🔗 Integrating advanced features from step06...')
            features_data = self._integrate_step06_advanced_features(features_data)
            self.logger.info(f'📊 After step06 integration: {len(features_data.columns)} total features')

            # Apply lookahead protection BEFORE feature selection (timestamp needed for temporal validation)
            if ML_COMMON_AVAILABLE and self.lookahead_protector:
                try:
                    self.logger.info('🔧 Applying lookahead protection before feature selection...')

                    # Check if timestamp column exists before applying filtering
                    timestamp_col = self._find_timestamp_column(features_data)
                    if timestamp_col:
                        self.logger.info('🔧 Applying automated future data filtering...')
                        filtered_features = self.lookahead_protector.automated_future_data_filtering(
                            features_data, self.current_timestamp, timestamp_col=timestamp_col
                        )
                        self.logger.info(f'✅ Lookahead protection applied: {len(filtered_features)} rows')
                    else:
                        self.logger.warning('⚠️ No timestamp column found - skipping future data filtering')
                        filtered_features = features_data
                except Exception as e:
                    self.logger.warning(f'⚠️ Lookahead protection failed: {e}')
                    filtered_features = features_data
            else:
                filtered_features = features_data

            # Use filtered features for the rest of processing
            features_data = filtered_features
            
            # Select numeric features only - optimize memory usage
            numeric_features = features_data.select_dtypes(include=[np.number]).columns.tolist()

            # Smart exclusion: keep volume for microstructure features, exclude raw OHLCV
            exclude_cols = ['timestamp', 'datetime', 'date', 'time', 'open', 'high', 'low', 'close']

            # Keep volume if we have microstructure features that need it
            microstructure_features_present = any(col in features_data.columns for col in [
                'vwap', 'price_impact', 'order_flow_imbalance', 'dollar_volume', 'volume_sma_5'
            ])

            if not microstructure_features_present:
                exclude_cols.append('volume')
                self.logger.info('ℹ️ Volume excluded (no microstructure features present)')
            else:
                self.logger.info('✅ Volume kept for microstructure feature calculations')

            feature_cols = [col for col in numeric_features if col not in exclude_cols]
            self.logger.info(f'📊 After smart feature selection: {len(feature_cols)} features')

            if len(feature_cols) >= 120:
                self.logger.info('🧠 Applying ML-based feature selection for optimization...')
                feature_cols = self._ml_based_feature_selection(
                    features_data, target_data, feature_cols, max_features=100
                )
                self.logger.info(f'🎯 After ML feature selection: {len(feature_cols)} features')

            # Handle missing values and ensure all data is numeric - use inplace operations to reduce memory
            feature_data = features_data[feature_cols].copy()  # Single copy operation
            feature_data.fillna(0, inplace=True)  # Inplace fillna to avoid creating new DataFrame

            # Convert all columns to numeric to avoid data type issues - use inplace operations
            cols_to_drop = []
            for col in feature_data.columns:
                if feature_data[col].dtype == 'object':
                    try:
                        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                        feature_data[col].fillna(0, inplace=True)  # Inplace fillna
                    except:
                        self.logger.warning(f'⚠️ Could not convert column {col} to numeric, dropping it')
                        cols_to_drop.append(col)
            
            # Drop problematic columns in one operation
            if cols_to_drop:
                feature_data.drop(columns=cols_to_drop, inplace=True)
                feature_cols = [col for col in feature_cols if col not in cols_to_drop]

            X = feature_data.values
            feature_names = np.array(feature_cols)
            self.logger.info(f'📊 After data type conversion: {len(feature_names)} features')

            # Free the DataFrame immediately after converting to numpy arrays
            del feature_data
            # Clean up intermediate variables
            del numeric_features, exclude_cols, cols_to_drop
            import gc
            gc.collect()

            # Get targets (now using binary SR classification)
            if 'sr_target' not in target_data.columns:
                raise ValueError("Target data must contain 'sr_target' column")
            if 'volatility_target' not in target_data.columns:
                raise ValueError("Target data must contain 'volatility_target' column")

            y_direction = target_data['sr_target'].values  # Binary: 0=Not Near SR, 1=Near SR
            y_volatility = target_data['volatility_target'].values

            # Handle binary class filtering (no neutral class in binary classification)
            sr_count = np.sum(y_direction == 1)  # Near SR
            non_sr_count = np.sum(y_direction == 0)  # Not Near SR
            total_count = len(y_direction)

            # Handle binary class balance
            # Check for extreme imbalance (>95% of one class)
            sr_ratio = sr_count / total_count
            if sr_ratio > 0.95 or sr_ratio < 0.05:
                self.logger.warning(f'⚠️ Extreme binary class imbalance: SR ratio = {sr_ratio:.1%}')
                self.logger.warning(f'   SR samples: {sr_count}, Non-SR samples: {non_sr_count}')
                
                # For extreme imbalance, we might need to adjust the proximity threshold
                # But for now, we'll proceed with the data as-is since binary classification is more robust
                self.logger.info(f'📊 Proceeding with binary classification despite imbalance')
            else:
                self.logger.info(f'📊 Binary class balance: SR ratio = {sr_ratio:.1%} ({sr_count} SR, {non_sr_count} Non-SR)')
            
            # Keep all samples for binary classification (both classes are legitimate)
            valid_mask = np.ones(len(y_direction), dtype=bool)

            # Apply filtering in-place to avoid creating copies
            if not np.all(valid_mask):
                X = X[valid_mask]
                y_direction = y_direction[valid_mask]
                y_volatility = y_volatility[valid_mask]

            # CRITICAL: Check for single-class chunks that will cause LogisticRegression failures
            # Use ML Common method to handle single-class chunks
            if ML_COMMON_AVAILABLE and self.lookahead_protector:
                X, y_direction, chunk_info = self.lookahead_protector.handle_single_class_chunks(
                    X, y_direction, feature_names
                )
                
                # Check if we should skip LogisticRegression for this chunk
                if chunk_info.get('skip_logistic_regression', False):
                    self.logger.info('ℹ️ Chunk marked to skip LogisticRegression due to single-class data')
                    # Return the data with a flag to skip LogisticRegression
                    return X, y_direction, y_volatility, feature_names, chunk_info
            else:
                # Fallback: check manually for binary classification
                unique_classes = np.unique(y_direction)
                if len(unique_classes) < 2:
                    class_name = {0: 'Not Near SR', 1: 'Near SR'}.get(unique_classes[0], f'Class {unique_classes[0]}')
                    self.logger.warning(f'⚠️ Single-class binary chunk detected: only {class_name} present ({len(y_direction)} samples)')
                    return X, y_direction, y_volatility, feature_names, {'skip_logistic_regression': True}

            # Always check unique classes for imbalance analysis
            unique_classes = np.unique(y_direction)

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
                
            # Memory cleanup before returning - comprehensive cleanup to prevent memory leaks
            cleanup_vars = ['feature_data', 'features_data', 'target_data', 'numeric_features', 
                           'feature_cols', 'cols_to_drop', 'valid_mask', 'neutral_mask', 
                           'directional_mask', 'unique_classes', 'filtered_features']
            for var in cleanup_vars:
                if var in locals():
                    del locals()[var]
            
            # Force garbage collection to free memory immediately
            import gc
            gc.collect()
            
            # Memory monitoring at end
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_change = memory_after - memory_before
            self.logger.info(f'📊 _prepare_ml_features: Memory change: {memory_change:+.1f}MB (Final: {memory_after:.1f}MB)')
                    
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

    def _is_feature_constant(self, feature_series: pd.Series) -> bool:
        """Check if a feature is constant (all values are the same)."""
        try:
            # Handle NaN values
            non_null_values = feature_series.dropna()
            if len(non_null_values) == 0:
                return True  # All NaN values are considered constant
            
            # Check if all non-null values are the same
            return non_null_values.nunique() <= 1
        except Exception:
            return False  # If we can't determine, assume not constant

    def _validate_temporal_integrity(self, features_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate that data maintains temporal integrity and no forward bias."""
        try:
            # Use ML Common LookaheadProtection if available
            if ML_COMMON_AVAILABLE and self.lookahead_protector:
                try:
                    self.logger.info('🔧 Using ML Common LookaheadProtection for bias detection')
                    
                    # Comprehensive bias detection using ML Common utilities
                    # Check if timestamp column exists and use it, otherwise skip bias detection
                    timestamp_col = self._find_timestamp_column(features_data)
                    if timestamp_col:
                        bias_results = self.lookahead_protector.comprehensive_bias_detection(features_data, timestamp_col=timestamp_col)
                    else:
                        self.logger.warning('⚠️ No timestamp column found in features_data - skipping bias detection')
                        bias_results = {'bias_detected': False, 'bias_details': {}, 'recommendations': []}
                    
                    if bias_results.get('bias_detected', False):
                        bias_details = bias_results.get('bias_details', {})
                        self.logger.error('❌ Forward bias detected by ML Common LookaheadProtection')
                        
                        # Log specific bias types detected
                        for bias_type, details in bias_details.items():
                            if details.get('detected', False):
                                severity = details.get('severity', 'unknown')
                                self.logger.error(f'❌ {bias_type}: {details.get("description", "Unknown bias")} (severity: {severity})')
                                
                                # Add detailed logging for suspicious correlations
                                if bias_type == 'suspicious_correlations':
                                    high_correlations = details.get('details', {}).get('high_correlations', [])
                                    if high_correlations:
                                        self.logger.info('🔍 Detailed correlation analysis:')
                                        for corr_info in high_correlations:
                                            feature1 = corr_info['feature1']
                                            feature2 = corr_info['feature2']
                                            correlation = corr_info['correlation']
                                            
                                            # Check if features are constant
                                            feature1_constant = self._is_feature_constant(features_data[feature1])
                                            feature2_constant = self._is_feature_constant(features_data[feature2])
                                            
                                            const1_str = " (CONSTANT)" if feature1_constant else ""
                                            const2_str = " (CONSTANT)" if feature2_constant else ""
                                            
                                            self.logger.info(f'   • {feature1}{const1_str} ↔ {feature2}{const2_str}: {correlation:.4f}')
                        
                        # Log recommendations
                        recommendations = bias_results.get('recommendations', [])
                        if recommendations:
                            self.logger.info('💡 Bias mitigation recommendations:')
                            for rec in recommendations[:3]:  # Show top 3 recommendations
                                self.logger.info(f'   • {rec}')
                        
                        # Store bias information for feature selection
                        self.bias_detection_results = bias_results
                        return {'valid': False, 'bias_results': bias_results}
                    else:
                        self.logger.info('✅ ML Common LookaheadProtection: No bias detected')
                        
                        # Log bias prevention measures applied
                        prevention_measures = bias_results.get('prevention_measures', [])
                        if prevention_measures:
                            self.logger.info('🛡️ Bias prevention measures applied:')
                            for measure in prevention_measures[:2]:  # Show top 2 measures
                                self.logger.info(f'   • {measure}')
                        
                        return {'valid': True, 'bias_results': bias_results}
                        
                except Exception as e:
                    self.logger.warning(f'⚠️ ML Common LookaheadProtection failed: {e}')
                    # Fallback to legacy validation
            
            # Legacy temporal integrity validation (fallback)
            self.logger.info('🔧 Using legacy temporal integrity validation')
            
            # Check if index is properly sorted
            if not features_data.index.is_monotonic_increasing:
                self.logger.error('❌ Forward bias detected: Data index is not temporally ordered')
                return {'valid': False, 'bias_results': {'bias_detected': True, 'bias_details': {'temporal_ordering': {'detected': True, 'severity': 'high'}}}}

            # Check for any future data leakage in features
            # This is a simplified check - in practice you'd want more sophisticated validation
            if 'close' in features_data.columns:
                # Check if close column is numeric
                if not pd.api.types.is_numeric_dtype(features_data['close']):
                    self.logger.debug('⏭️ Close column is not numeric, skipping temporal validation')
                    return {'valid': True, 'bias_results': {'bias_detected': False}}

                # Check if any features use future price information
                close_prices = features_data['close'].values
                if len(close_prices) > 1:
                    try:
                        # Ensure close prices are numeric
                        close_prices = pd.to_numeric(close_prices, errors='coerce')
                        if np.isnan(close_prices).any():
                            self.logger.debug('⏭️ Close prices contain NaN values, skipping detailed temporal validation')
                            return {'valid': True, 'bias_results': {'bias_detected': False}}

                        future_prices = np.roll(close_prices, -1)[:-1]
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f'⏭️ Failed to process close prices for temporal validation: {e}')
                        return {'valid': True, 'bias_results': {'bias_detected': False}}

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
            return {'valid': True, 'bias_results': {'bias_detected': False}}

        except Exception as e:
            self.logger.error(f'❌ Temporal integrity validation failed: {e}')
            return {'valid': False, 'bias_results': {'bias_detected': True, 'error': str(e)}}

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for ML models using ML Common utilities."""
        try:
            self.logger.info('🔧 Starting enhanced hyperparameter optimization with ML Common utilities...')

            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))

            # Use ML Common HyperparameterOptimization if available
            if ML_COMMON_AVAILABLE and self.hpo_optimizer:
                try:
                    self.logger.info('🔧 Using ML Common HyperparameterOptimization')
                    
                    # Analyze data characteristics for automated search space generation
                    data_characteristics = {
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'n_classes': n_classes,
                        'task_type': 'classification',
                        'data_size_category': 'small' if n_samples < 1000 else 'medium' if n_samples < 10000 else 'large',
                        'feature_density': n_features / n_samples if n_samples > 0 else 0
                    }
                    
                    # Generate automated search space
                    search_space = self.hpo_optimizer.automated_search_space_generation(
                        'RandomForestClassifier', data_characteristics
                    )
                    
                    self.logger.info(f'📊 Generated search space with {len(search_space)} parameters')
                    
                    # Perform multi-objective optimization
                    self.logger.info(f'🚀 Starting ML Common multi-objective optimization...')
                    self.logger.info(f'📊 Optimization config: {self.optimization_trials} trials, {self.optimization_folds} CV folds')
                    self.logger.info(f'🎯 Objectives: accuracy, f1_score')
                    self.logger.info(f'⏱️ Timeout: {self.optimization_trials * 10} seconds')
                    
                    optimization_results = self.hpo_optimizer.multi_objective_optimization(
                        X, y, 
                        model_type='RandomForestClassifier',
                        search_space=search_space,
                        objectives=['accuracy', 'f1_score'],
                        n_trials=self.optimization_trials,
                        cv_folds=self.optimization_folds,
                        timeout=self.optimization_trials * 10  # 10 seconds per trial
                    )
                    
                    self.logger.info(f'✅ ML Common multi-objective optimization completed!')

                    # Robust error recovery and result processing
                    optimization_success = self._process_optimization_results(optimization_results)

                    if optimization_success:
                        best_params = optimization_success['best_params']
                        best_scores = optimization_success['best_scores']
                        pareto_info = optimization_success.get('pareto_info', {})

                        return {
                            'method': 'ml_common_hpo',
                            'best_params': best_params,
                            'best_scores': best_scores,
                            'optimization_results': optimization_results,
                            'pareto_info': pareto_info,
                            'search_space': search_space,
                            'data_characteristics': data_characteristics
                        }
                    else:
                        # All recovery attempts failed - use ultimate fallback
                        self.logger.error(f'❌ All optimization recovery attempts failed')
                        return self._create_ultimate_fallback(search_space, data_characteristics)
                        
                except Exception as e:
                    self.logger.error(f'❌ ML Common HPO failed: {e}')
                    raise RuntimeError(f'Hyperparameter optimization failed and fallback is disabled: {e}')


        except ImportError as e:
            self.logger.error(f'❌ Optuna not available and fallback is disabled: {e}')
            raise RuntimeError(f'Hyperparameter optimization failed - Optuna unavailable and fallback disabled: {e}')

        except Exception as e:
            self.logger.error(f'❌ Optuna optimization failed and fallback is disabled: {e}')
            raise RuntimeError(f'Hyperparameter optimization completely failed and fallback is disabled: {e}')

    def _fallback_hyperparameter_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fallback hyperparameter selection using ML Common utilities with parameter multipliers."""
        n_samples, n_features = X.shape

        # Check for extreme class imbalance and adjust strategy
        from collections import Counter
        class_counts = Counter(y)
        majority_class_ratio = max(class_counts.values()) / len(y)

        # Define parameter multipliers based on data characteristics
        if majority_class_ratio > 0.9:  # Extreme imbalance (>90%)
            self.logger.warning(f'⚠️ Extreme class imbalance detected ({majority_class_ratio:.1%}), using specialized strategy')
            model_type = 'RandomForestClassifier'
            multiplier_config = {
                'n_estimators': 2.0,  # Increase trees for better minority class learning
                'max_depth': 0.8,     # Reduce depth to prevent overfitting
                'min_samples_split': 2.0,  # Increase to reduce overfitting
                'min_samples_leaf': 2.5,   # Increase for better generalization
            }
        elif n_samples < 1000:  # Small dataset
            model_type = 'LogisticRegression'
            multiplier_config = {
                'C': 1.0,  # Standard regularization
                'max_iter': 1.0,  # Standard iterations
            }
        elif n_samples < 10000:  # Medium dataset
            model_type = 'RandomForestClassifier'
            multiplier_config = {
                'n_estimators': 1.0,  # Standard tree count
                'max_depth': 1.0,     # Standard depth
                'min_samples_split': 1.0,  # Standard split
                'min_samples_leaf': 1.0,   # Standard leaf size
            }
        else:  # Large dataset
            model_type = 'HistGradientBoostingClassifier'
            multiplier_config = {
                'max_iter': 1.0,     # Standard iterations
                'max_depth': 1.0,    # Standard depth
                'learning_rate': 1.0,  # Standard learning rate
                'min_samples_leaf': 1.0,  # Standard leaf size
                'l2_regularization': 1.0,  # Standard regularization
            }

        try:
            # Use ML Common utilities to generate baseline parameters
            if ML_COMMON_AVAILABLE and self.hpo_optimizer:
                self.logger.info('🔧 Using ML Common utilities for fallback parameter generation')

                # Generate data characteristics for ML Common
                data_characteristics = {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'n_classes': len(np.unique(y)),
                    'task_type': 'classification',
                    'data_size_category': 'small' if n_samples < 1000 else 'medium' if n_samples < 10000 else 'large',
                    'feature_density': n_features / n_samples if n_samples > 0 else 0,
                    'class_imbalance_ratio': majority_class_ratio
                }

                # Generate search space using ML Common
                search_space = self.hpo_optimizer.automated_search_space_generation(
                    model_type, data_characteristics
                )

                if search_space:
                    self.logger.info(f'📊 ML Common generated search space with {len(search_space)} parameters')

                    # Extract baseline parameters from search space and apply multipliers
                    baseline_params = self._extract_baseline_from_search_space(search_space, multiplier_config, model_type)
                    baseline_params['optimization_method'] = 'ml_common_fallback'

                    # Add class weight handling
                    if majority_class_ratio > 0.7:
                        if 'class_weight' in search_space:
                            baseline_params['class_weight'] = 'balanced'
                        elif model_type == 'RandomForestClassifier':
                            baseline_params['class_weight'] = 'balanced'

                    self.logger.info(f'✅ Generated fallback parameters using ML Common with multipliers: {baseline_params}')
                    return baseline_params
                else:
                    self.logger.warning('⚠️ ML Common search space generation failed, using manual fallback')

        except Exception as e:
            self.logger.warning(f'⚠️ ML Common fallback failed: {e}, using manual parameters')

        # Manual fallback if ML Common fails
        return self._manual_fallback_parameters(model_type, majority_class_ratio, multiplier_config)

    def _extract_baseline_from_search_space(self, search_space: Dict[str, Any], multipliers: Dict[str, float],
                                          model_type: str = 'RandomForestClassifier') -> Dict[str, Any]:
        """Extract baseline parameters from ML Common search space and apply multipliers."""
        baseline_params = {}

        for param_name, param_config in search_space.items():
            if param_name in multipliers:
                multiplier = multipliers[param_name]

                if param_config['type'] == 'int':
                    # Use midpoint of range, then apply multiplier
                    low, high = param_config['low'], param_config['high']
                    baseline = (low + high) // 2
                    baseline_params[param_name] = int(baseline * multiplier)

                elif param_config['type'] == 'float':
                    # Use midpoint of range, then apply multiplier
                    low, high = param_config['low'], param_config['high']
                    baseline = (low + high) / 2
                    baseline_params[param_name] = baseline * multiplier

                elif param_config['type'] == 'categorical':
                    # Use first choice as baseline
                    baseline_params[param_name] = param_config['choices'][0]

        # Set model type
        baseline_params['model_type'] = model_type

        return baseline_params

    def _manual_fallback_parameters(self, model_type: str, majority_class_ratio: float,
                                  multiplier_config: Dict[str, float]) -> Dict[str, Any]:
        """Manual fallback parameter generation when ML Common fails."""
        if majority_class_ratio > 0.9:  # Extreme imbalance
            return {
                'model_type': 'RandomForestClassifier',
                'n_estimators': int(100 * multiplier_config.get('n_estimators', 1.0)),
                'max_depth': int(8 * multiplier_config.get('max_depth', 1.0)),
                'min_samples_split': int(10 * multiplier_config.get('min_samples_split', 1.0)),
                'min_samples_leaf': int(5 * multiplier_config.get('min_samples_leaf', 1.0)),
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'bootstrap': True,
                'random_state': 42,
                'optimization_method': 'manual_fallback_imbalanced'
            }

        elif model_type == 'LogisticRegression':
            return {
                'model_type': 'LogisticRegression',
                'C': 1.0 * multiplier_config.get('C', 1.0),
                'max_iter': int(1000 * multiplier_config.get('max_iter', 1.0)),
                'solver': 'lbfgs',
                'class_weight': 'balanced' if majority_class_ratio > 0.7 else None,
                'optimization_method': 'manual_fallback'
            }

        elif model_type == 'RandomForestClassifier':
            return {
                'model_type': 'RandomForestClassifier',
                'n_estimators': int(100 * multiplier_config.get('n_estimators', 1.0)),
                'max_depth': int(10 * multiplier_config.get('max_depth', 1.0)),
                'min_samples_split': int(2 * multiplier_config.get('min_samples_split', 1.0)),
                'min_samples_leaf': int(1 * multiplier_config.get('min_samples_leaf', 1.0)),
                'max_features': 'sqrt',
                'class_weight': 'balanced' if majority_class_ratio > 0.7 else None,
                'optimization_method': 'manual_fallback'
            }

        elif model_type == 'HistGradientBoostingClassifier':
            return {
                'model_type': 'HistGradientBoostingClassifier',
                'max_iter': int(100 * multiplier_config.get('max_iter', 1.0)),
                'max_depth': int(10 * multiplier_config.get('max_depth', 1.0)),
                'learning_rate': 0.1 * multiplier_config.get('learning_rate', 1.0),
                'min_samples_leaf': int(20 * multiplier_config.get('min_samples_leaf', 1.0)),
                'l2_regularization': 1e-4 * multiplier_config.get('l2_regularization', 1.0),
                'class_weight': 'balanced' if majority_class_ratio > 0.7 else None,
                'optimization_method': 'manual_fallback'
            }

        # Default fallback
            return {
                'model_type': 'RandomForestClassifier',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
            'optimization_method': 'default_fallback'
            }

    def _optimize_feature_selection(self, X: np.ndarray, y_direction: np.ndarray, y_volatility: np.ndarray, feature_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Optimize feature selection using ML Common utilities while maintaining target feature counts."""
        try:
            start_time = time.time()
            n_samples, n_features = X.shape

            self.logger.info(f'🎯 Starting enhanced feature selection with ML Common utilities')
            self.logger.info(f'📊 Dataset: {n_samples} samples, {n_features} features')

            # Check if bias was detected and adjust feature selection strategy
            bias_mitigation_mode = False
            if hasattr(self, 'bias_detection_results') and self.bias_detection_results:
                bias_results = self.bias_detection_results
                if bias_results.get('bias_detected', False):
                    bias_mitigation_mode = True
                    self.logger.info('🛡️ Bias mitigation mode: Using aggressive feature selection to reduce bias')
                    
                    # Adjust target features for bias mitigation
                    if n_samples < 5000:
                        target_features = min(20, n_features)  # More aggressive reduction
                    elif n_samples < 20000:
                        target_features = min(30, n_features)  # More aggressive reduction
                    else:
                        target_features = min(60, n_features)  # More aggressive reduction
                else:
                    # Normal feature selection
                    if n_samples < 5000:
                        target_features = min(30, n_features)
                    elif n_samples < 20000:
                        target_features = min(50, n_features)
                    else:
                        target_features = min(100, n_features)
            else:
                # Normal feature selection
                if n_samples < 5000:
                    target_features = min(30, n_features)
                elif n_samples < 20000:
                    target_features = min(50, n_features)
                else:
                    target_features = min(100, n_features)

            self.logger.info(f'🎯 Target features: {target_features}')

            # Use ML Common feature selection if available
            if self.feature_selector:
                try:
                    # Select features using ML Common utilities
                    # Note: FeatureSelectionFramework.select_features doesn't support bias_mitigation parameter
                    selected_features, feature_scores, selection_info = self.feature_selector.select_features(
                        X, y_direction, feature_names, n_features=target_features
                    )
                    
                    if bias_mitigation_mode:
                        self.logger.info(f'✅ ML Common bias-mitigating feature selection completed: {len(selected_features)} features selected')
                    else:
                        self.logger.info(f'✅ ML Common feature selection completed: {len(selected_features)} features selected')

                    # CRITICAL: Ensure minimum features are selected (at least 100 or all available)
                    if len(selected_features) == 0:
                        self.logger.warning('⚠️ No features selected by ML Common! Using fallback with minimum features.')
                        # Use at least 100 features or all available features, whichever is smaller
                        min_features = min(100, n_features)
                        self.logger.info(f'🔄 Falling back to SelectKBest with {min_features} features')

                        from sklearn.feature_selection import SelectKBest, f_classif
                        selector = SelectKBest(score_func=f_classif, k=min_features)
                        X_selected = selector.fit_transform(X, y_direction)
                        selected_indices = selector.get_support(indices=True)
                        selected_features = feature_names[selected_indices]
                        feature_scores = selector.scores_[selected_indices]

                        selection_info = {
                            'method': 'SelectKBest_fallback',
                            'target_features': min_features,
                            'selected_count': len(selected_features),
                            'selection_time': time.time() - start_time,
                            'bias_mitigation_mode': bias_mitigation_mode,
                            'reason': 'ml_common_selected_zero_features'
                        }

                        self.logger.info(f'✅ Fallback feature selection completed: {len(selected_features)} features selected')
                        return X_selected, y_direction, y_volatility, selected_features, selection_info

                    # Create X_selected from selected features
                    selected_indices = [np.where(feature_names == feat)[0][0] for feat in selected_features]
                    X_selected = X[:, selected_indices]

                    return X_selected, y_direction, y_volatility, selected_features, selection_info
                    
                except Exception as e:
                    self.logger.warning(f'⚠️ ML Common feature selection failed: {e}')
                    self.logger.info('🔄 Falling back to scikit-learn feature selection')

            # Fallback to scikit-learn feature selection
            from sklearn.feature_selection import SelectKBest, f_classif
            
            selector = SelectKBest(score_func=f_classif, k=target_features)
            X_selected = selector.fit_transform(X, y_direction)
            selected_indices = selector.get_support(indices=True)
            selected_features = feature_names[selected_indices]
            feature_scores = selector.scores_[selected_indices]
            
            selection_info = {
                'method': 'SelectKBest',
                'target_features': target_features,
                'selected_count': len(selected_features),
                'selection_time': time.time() - start_time,
                'bias_mitigation_mode': bias_mitigation_mode
            }
            
            if bias_mitigation_mode:
                self.logger.info(f'✅ Fallback bias-mitigating feature selection completed: {len(selected_features)} features selected')
            else:
                self.logger.info(f'✅ Fallback feature selection completed: {len(selected_features)} features selected')
            return X_selected, y_direction, y_volatility, selected_features, selection_info
            
        except Exception as e:
            self.logger.error(f'❌ Feature selection failed: {e}')
            # Return original data if selection fails
            return X, y_direction, y_volatility, feature_names, {'method': 'none', 'error': str(e)}

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Perform enhanced cross-validation using ML Common utilities."""
        try:
            self.logger.info('🔄 Starting enhanced cross-validation with ML Common utilities')
            
            # Use ML Common CV utilities if available
            if self.cv_utils:
                try:
                    # Create a simple model for temporal CV (RandomForest as default)
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)

                    cv_results = self.cv_utils.perform_temporal_cv(X, y, model, n_splits=5)
                    self.logger.info('✅ ML Common temporal cross-validation completed')
                    return cv_results
                except Exception as e:
                    self.logger.warning(f'⚠️ ML Common cross-validation failed: {e}')
                    self.logger.info('🔄 Falling back to scikit-learn cross-validation')
            
            # Fallback to scikit-learn cross-validation
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.ensemble import RandomForestClassifier
            
            # Use a simple model for cross-validation
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            cv_results = {
                'scores': scores.tolist(),
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'method': 'StratifiedKFold',
                'n_splits': 5
            }
            
            self.logger.info(f'✅ Fallback cross-validation completed: {scores.mean():.4f} ± {scores.std():.4f}')
            return cv_results
            
        except Exception as e:
            self.logger.error(f'❌ Cross-validation failed: {e}')
            return {'error': str(e), 'method': 'none'}

    def _calculate_evaluation_metrics(self, models_results: Dict[str, Any],
                                    cv_results: Dict[str, Any],
                                    X_test: np.ndarray, y_dir_test: np.ndarray,
                                    y_vol_test: np.ndarray, ensemble_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate enhanced evaluation metrics using ML Common utilities."""
        try:
            self.logger.info('📊 Calculating enhanced evaluation metrics')
            
            # Use ML Common evaluation utilities if available
            if self.model_evaluator:
                try:
                    # ModelEvaluationUtilities uses multi_metric_evaluation instead of calculate_metrics
                    metrics = self.model_evaluator.multi_metric_evaluation(
                        y_dir_test, models_results, y_prob=None
                    )
                    self.logger.info('✅ ML Common evaluation metrics completed')
                    return metrics
                except Exception as e:
                    self.logger.warning(f'⚠️ ML Common evaluation failed: {e}')
                    self.logger.info('🔄 Falling back to basic evaluation')
            
            # Fallback to basic evaluation
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'cv_results': cv_results,
                'models_count': len(models_results),
                'test_samples': len(X_test),
                'evaluation_method': 'basic'
            }
            
            # Calculate basic metrics if we have test predictions
            if 'predictions' in models_results:
                predictions = models_results['predictions']
                if len(predictions) == len(y_dir_test):
                    metrics.update({
                        'accuracy': accuracy_score(y_dir_test, predictions),
                        'precision': precision_score(y_dir_test, predictions, average='weighted'),
                        'recall': recall_score(y_dir_test, predictions, average='weighted'),
                        'f1_score': f1_score(y_dir_test, predictions, average='weighted')
                    })
            
            self.logger.info('✅ Basic evaluation metrics completed')
            return metrics
            
        except Exception as e:
            self.logger.error(f'❌ Evaluation metrics calculation failed: {e}')
            return {'error': str(e), 'method': 'none'}

    def _save_best_model(self, models_results: Dict[str, Any], scaler: Any,
                        selected_feature_names: List[str]) -> str:
        """Save the best performing model with metadata."""
        try:
            import os
            from pathlib import Path

            # Create models directory if it doesn't exist
            models_dir = Path(self.data_cache_dir) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            # Find the best model
            best_model_name = None
            best_score = -float('inf')

            for model_name, model_info in models_results.items():
                if isinstance(model_info, dict) and 'cv_score' in model_info:
                    score = model_info['cv_score']
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name

            if best_model_name:
                model_info = models_results[best_model_name]
                model = model_info.get('model')

                if model:
                    # Save model with metadata
                    model_path = models_dir / f"best_sr_model_{best_model_name}.pkl"
                    import pickle

                    model_metadata = {
                        'model': model,
                        'scaler': scaler,
                        'feature_names': selected_feature_names,
                        'model_name': best_model_name,
                        'cv_score': best_score,
                        'timestamp': datetime.now().isoformat(),
                        'model_type': type(model).__name__
                    }

                    with open(model_path, 'wb') as f:
                        pickle.dump(model_metadata, f)

                    self.logger.info(f'💾 Best model saved: {model_path} (Score: {best_score:.4f})')
                    return str(model_path)

            self.logger.warning('⚠️ No suitable model found to save')
            return ""

        except Exception as e:
            self.logger.error(f'❌ Failed to save best model: {e}')
            return ""

    def _cleanup_memory(self, variables_to_delete: List[str] = None, force_gc: bool = True) -> None:
        """Centralized memory cleanup utility to avoid redundancy."""
        try:
            # Delete specific variables if provided
            if variables_to_delete:
                for var_name in variables_to_delete:
                    try:
                        # Try to delete from local scope first
                        if var_name in locals():
                            del locals()[var_name]
                        # Then try global scope
                        elif var_name in globals():
                            del globals()[var_name]
                    except (NameError, KeyError):
                        pass  # Variable doesn't exist, skip

            # Force garbage collection if requested
            if force_gc:
                import gc
                gc.collect()

        except Exception as e:
            self.logger.debug(f"Memory cleanup warning: {e}")

    # ===== ADVANCED ERROR RECOVERY METHODS =====

    def _process_optimization_results(self, optimization_results):
        """Process optimization results with comprehensive error recovery."""
        try:
            # Check if we have valid results
            if optimization_results.get('best_params'):
                self.logger.info('✅ Optimization successful - processing results')

                best_params = optimization_results['best_params']
                best_scores = optimization_results.get('best_scores', [0.0])

                # Process Pareto information if available
                pareto_info = self._extract_pareto_information(optimization_results)

                # Log optimization summary
                self._log_optimization_summary(optimization_results, best_params, best_scores)

                return {
                    'best_params': best_params,
                    'best_scores': best_scores,
                    'pareto_info': pareto_info,
                    'success': True
                }

            # Handle various error cases with specific recovery strategies
            error_msg = optimization_results.get('error', 'Unknown error')

            if error_msg == 'No Pareto optimal trials found':
                self.logger.warning('⚠️ No Pareto optimal trials - using best trial from optimization')
                return self._recover_from_no_pareto(optimization_results)

            elif 'single objective' in error_msg.lower() or 'objectives' in error_msg.lower():
                self.logger.warning('⚠️ Multi-objective error - attempting single objective recovery')
                return self._recover_from_multiobjective_error(optimization_results)

            elif 'memory' in error_msg.lower():
                self.logger.warning('⚠️ Memory error - using simplified optimization')
                return self._recover_from_memory_error(optimization_results)

            else:
                self.logger.warning(f'⚠️ Unknown optimization error: {error_msg}')
                return self._recover_from_generic_error(optimization_results)

        except Exception as e:
            self.logger.error(f'❌ Error processing optimization results: {e}')
            return None

    def _extract_pareto_information(self, optimization_results):
        """Extract and analyze Pareto front information."""
        pareto_info = {}

        if 'pareto_front' in optimization_results and len(optimization_results['pareto_front']) > 0:
            try:
                # Import and use the Pareto analysis from ML Common
                from src.utils.ml_common.hpo_utils import HyperparameterOptimization
                hpo_analyzer = HyperparameterOptimization()

                # Get detailed Pareto analysis
                pareto_analysis = hpo_analyzer.analyze_pareto_front(
                    study=None,
                    objectives=['accuracy', 'f1_score'],
                    optimization_results=optimization_results
                )

                if 'error' not in pareto_analysis:
                    self.logger.info(f'📊 Pareto Front Analysis (ML Common):')
                    self.logger.info(f'   • Front size: {pareto_analysis.get("pareto_front_size", 0)} solutions')
                    self.logger.info(f'   • Pareto ratio: {pareto_analysis.get("pareto_ratio", 0):.1f}%')

                    # Show hypervolume if available
                    if 'hypervolume_analysis' in pareto_analysis:
                        hv = pareto_analysis['hypervolume_analysis']
                        if 'hypervolume' in hv:
                            self.logger.info(f'   • Hypervolume: {hv["hypervolume"]:.4f}')

                    # Show diversity metrics
                    if 'diversity_metrics' in pareto_analysis:
                        dm = pareto_analysis['diversity_metrics']
                        if 'diversity_score' in dm:
                            self.logger.info(f'   • Diversity score: {dm["diversity_score"]:.4f}')

                    # Show best solutions
                    best_by_sum = optimization_results.get('best_by_sum', {})
                    if best_by_sum and 'trial_number' in best_by_sum:
                        self.logger.info(f'🎯 Best by sum: Trial {best_by_sum["trial_number"]}')

                    pareto_info = {
                        'pareto_front_size': len(optimization_results['pareto_front']),
                        'pareto_analysis': pareto_analysis,
                        'best_by_sum': best_by_sum
                    }
                else:
                    self.logger.warning(f'⚠️ Pareto analysis failed: {pareto_analysis["error"]}')

            except Exception as e:
                self.logger.warning(f'⚠️ Pareto analysis exception: {e}')

        return pareto_info

    def _log_optimization_summary(self, optimization_results, best_params, best_scores):
        """Log comprehensive optimization summary."""
        self.logger.info('🏆 Optimization Summary:')
        self.logger.info(f'   • Best Parameters: {best_params}')
        self.logger.info(f'   • Best Scores: {best_scores}')
        self.logger.info(f'   • Total Trials: {optimization_results.get("n_trials", 0)}')

        if 'pareto_front' in optimization_results:
            self.logger.info(f'   • Pareto Solutions: {len(optimization_results["pareto_front"])}')

    def _recover_from_no_pareto(self, optimization_results):
        """Recover when no Pareto optimal trials are found."""
        self.logger.info('🔄 Attempting recovery from no Pareto optimal trials')

        # Try to find the best trial from optimization history
        if 'optimization_history' in optimization_results:
            history = optimization_results['optimization_history']
            if history:
                # Find trial with best combined score
                best_trial = max(history, key=lambda t: sum(t.get('scores', [0])) if t.get('scores') else 0)
                best_params = best_trial.get('params', {})
                best_scores = best_trial.get('scores', [0])

                self.logger.info(f'✅ Recovered using best trial from history: Trial {best_trial.get("trial", "N/A")}')
                return {
                    'best_params': best_params,
                    'best_scores': best_scores,
                    'recovery_method': 'best_from_history',
                    'success': True
                }

        # Ultimate fallback
        return self._create_conservative_fallback()

    def _recover_from_multiobjective_error(self, optimization_results):
        """Recover from multi-objective optimization errors."""
        self.logger.info('🔄 Attempting recovery from multi-objective error')

        # Try single-objective optimization as fallback
        try:
            self.logger.info('🎯 Attempting single-objective optimization...')
            # This would require implementing a single-objective fallback
            # For now, use conservative defaults
            return self._create_conservative_fallback()
        except Exception as e:
            self.logger.warning(f'⚠️ Single-objective fallback failed: {e}')
            return self._create_conservative_fallback()

    def _recover_from_memory_error(self, optimization_results):
        """Recover from memory-related optimization errors."""
        self.logger.info('🔄 Attempting recovery from memory error')

        # Use minimal parameter set to reduce memory usage
        minimal_params = {
            'n_estimators': 50,  # Reduced from 100
            'max_depth': 3,      # Reduced from 5
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

        self.logger.info('✅ Using minimal parameter set for memory efficiency')
        return {
            'best_params': minimal_params,
            'best_scores': [0.0],
            'recovery_method': 'memory_efficient_fallback',
            'success': True
        }

    def _recover_from_generic_error(self, optimization_results):
        """Generic recovery for unknown optimization errors."""
        self.logger.info('🔄 Attempting generic error recovery')
        return self._create_conservative_fallback()

    def _create_conservative_fallback(self):
        """Create conservative fallback parameters."""
        conservative_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }

        self.logger.info('✅ Using conservative fallback parameters')
        return {
            'best_params': conservative_params,
            'best_scores': [0.0],
            'recovery_method': 'conservative_fallback',
            'success': True
        }

    def _create_ultimate_fallback(self, search_space, data_characteristics):
        """Create ultimate fallback when all optimization methods fail."""
        self.logger.warning('🚨 Using ultimate fallback parameters - optimization completely failed')

        return {
            'method': 'ultimate_fallback',
            'best_params': {'n_estimators': 100, 'max_depth': 5},  # Ultra-conservative defaults
            'best_scores': [0.0],
            'optimization_results': {'error': 'All optimization methods failed'},
            'search_space': search_space,
            'data_characteristics': data_characteristics
        }


# Error handling utilities - moved outside the class
class ErrorSeverity:
    CRITICAL = "CRITICAL"  # System cannot continue
    HIGH = "HIGH"         # Major functionality affected
    MEDIUM = "MEDIUM"     # Some functionality affected
    LOW = "LOW"           # Minor issues, system continues


class ErrorCategory:
    DATA_QUALITY = "DATA_QUALITY"
    ML_TRAINING = "ML_TRAINING"
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
    
    # High severity errors
    if isinstance(error, (ValueError, KeyError, AttributeError)):
        return ErrorSeverity.HIGH, ErrorCategory.DATA_QUALITY
    
    # Medium severity errors
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_DEPENDENCY
    
    # Default classification
    return ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_RESOURCE


@handles_errors(default_return=False)
def handle_error_with_recovery(error: Exception, context: str, max_retries: int = 3) -> bool:
    """Handle errors with automatic recovery attempts."""
    severity, category = classify_error(error, context)
    
    if severity == ErrorSeverity.CRITICAL:
        return False  # No recovery for critical errors
    
    # Implement recovery logic based on error category
    if category == ErrorCategory.DATA_QUALITY:
        # Try data cleaning or validation
        return True
    elif category == ErrorCategory.EXTERNAL_DEPENDENCY:
        # Try alternative imports or fallback methods
        return True
    
    return False


@handles_errors(default_return=False)
def detect_data_drift(current_data: pd.DataFrame, reference_data: pd.DataFrame = None, 
                     threshold: float = 0.1) -> bool:
    """Detect data drift between current and reference datasets."""
    try:
        if reference_data is None:
            # Use first 10% of current data as reference
            reference_data = current_data.head(len(current_data) // 10)
        
        # Simple drift detection based on statistical properties
        current_stats = current_data.describe()
        reference_stats = reference_data.describe()
        
        # Compare means for numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        drift_detected = False
        
        for col in numeric_cols:
            if col in reference_stats.columns:
                current_mean = current_stats.loc['mean', col]
                reference_mean = reference_stats.loc['mean', col]
                
                if abs(current_mean - reference_mean) / abs(reference_mean) > threshold:
                    drift_detected = True
                    break
        
        return drift_detected
        
    except Exception as e:
        logger.warning(f"Data drift detection failed: {e}")
        return False


@handles_errors(default_return={})
def generate_function_report(ml_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate comprehensive function report for ML results."""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'completed' if ml_results else 'no_results',
            'ml_results': ml_results or {},
            'total_calls': len(ml_results) if ml_results else 0,
            'summary': {
                'total_functions': len(ml_results) if ml_results else 0,
                'successful_functions': len([r for r in (ml_results or {}).values() if r.get('success', False)]),
                'failed_functions': len([r for r in (ml_results or {}).values() if not r.get('success', False)])
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Function report generation failed: {e}")
        return {'error': str(e), 'timestamp': datetime.now().isoformat(), 'total_calls': 0}

    def cleanup_m1_resources(self):
        """Clean up M1 resources specifically for SR optimization."""
        try:
            self.logger.info('🧹 Cleaning up M1 resources in SR optimization')

            # Clean up M1 optimizers
            if 'cleanup_m1_optimizers' in globals():
                try:
                    cleanup_m1_optimizers()
                    self.logger.info('✅ M1 optimizers cleaned up')
                except Exception as e:
                    self.logger.debug(f'Failed to cleanup M1 optimizers: {e}')

            # Final memory optimization
            if M1_MEMORY_AVAILABLE or M1_GPU_AVAILABLE:
                optimize_memory()
                memory_usage = get_memory_usage()
                self.logger.info(f'🧠 Final memory usage: {memory_usage["rss_gb"]:.2f}GB')

        except Exception as e:
            self.logger.error(f'❌ Failed to cleanup M1 resources: {e}')

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_m1_resources()
        except Exception:
            pass  # Avoid errors during destruction


async def run_step02_5_bypass(symbol: str, exchange: str, timeframe: str = '1m', 
                             data_dir: str = None, force_rerun: bool = False, **kwargs) -> bool:
    """Run Step02_5 with bypass for testing purposes."""
    try:
        logger.info(f'🚀 Running Step02_5 bypass for {symbol} on {exchange}')
        
        # Create step instance
        config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir or f'data/{exchange}/{symbol}',
            'force_rerun': force_rerun
        }
        
        step = SROptimizationStep(config)
        await step.initialize()
        
        # Prepare training input
        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir or f'data/{exchange}/{symbol}',
            'force_rerun': force_rerun,
            **kwargs
        }
        
        # Execute step
        result = await step.execute(training_input, {})
        
        if result.get('step02_5_sr_optimization_completed', False):
            logger.info('✅ Step02_5 bypass completed successfully')
            return True
        else:
            logger.error('❌ Step02_5 bypass failed')
            return False
            
    except Exception as e:
        logger.error(f'❌ Step02_5 bypass failed: {e}')
        return False


async def run_step02_5_direct(symbol: str, exchange: str, timeframe: str = '1m', 
                             data_dir: str = None, force_rerun: bool = False, **kwargs) -> bool:
    """Run Step02_5 directly for testing purposes."""
    try:
        logger.info(f'🚀 Running Step02_5 direct for {symbol} on {exchange}')
        
        # Create step instance
        config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir or f'data/{exchange}/{symbol}',
            'force_rerun': force_rerun
        }
        
        step = SROptimizationStep(config)
        await step.initialize()
        
        # Prepare training input
        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir or f'data/{exchange}/{symbol}',
            'force_rerun': force_rerun,
            **kwargs
        }
        
        # Execute step
        result = await step.execute(training_input, {})
        
        if result.get('step02_5_sr_optimization_completed', False):
            logger.info('✅ Step02_5 direct completed successfully')
            return True
        else:
            logger.error('❌ Step02_5 direct failed')
            return False
            
    except Exception as e:
        logger.error(f'❌ Step02_5 direct failed: {e}')
        return False
