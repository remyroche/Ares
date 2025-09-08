"""
Step05 Dependency Injection Container

This module provides a comprehensive dependency injection container for Step05 utilities,
ensuring proper initialization and management of all utility dependencies.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from dataclasses import dataclass
from pathlib import Path

# Import all utility modules
from src.utils.common_operations import (
    get_current_datetime, get_today, format_datetime, parse_datetime,
    create_empty_dataframe, safe_fillna, safe_rolling, safe_copy, safe_deepcopy,
    safe_mean, safe_std, ensure_directory, safe_file_exists, safe_json_dump, 
    safe_json_load, safe_read_parquet, safe_to_parquet, safe_sleep, safe_gather,
    create_async_task, safe_append, safe_extend, safe_dict_get, safe_dict_items,
    safe_lower, safe_upper, safe_join, get_logger, setup_basic_logging,
    create_argument_parser, add_common_arguments, safe_exception_handler,
    safe_float, safe_int, suggest_float_uniform, suggest_int_uniform,
    validate_dataframe, validate_numeric_range, validate_dataframe_schema,
    validate_data_quality, optimize_dataframe_dtypes, timed_operation,
    format_bytes, chunked_iterable, parallel_map, safe_log_metric,
    safe_log_params, safe_log_artifact, standardize_price_action_probabilities
)

from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
    validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError
)

from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
    save_data, load_data, SerializationError
)

from src.utils.data_processing_utils import (
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
    validate_dataframe as validate_dataframe_advanced, clean_dataframe,
    transform_dataframe, get_dataframe_info as get_dataframe_info_advanced,
    DataQualityLevel, DataQualityIssue, DataQualityReport
)

from src.utils.m1_gpu_utils import (
    M1GPUManager, M1PerformanceOptimizer, create_m1_optimized_config,
    initialize_m1_gpu, get_m1_gpu_manager, m1_tensor_multiply, m1_batch_process,
    m1_monte_carlo_simulate
)

from src.utils.m1_memory_optimizer import (
    M1MemoryOptimizer, M1DataManager, get_m1_memory_optimizer,
    create_memory_efficient_dataframe, memory_efficient_groupby
)

from src.utils.m1_cpu_optimizer import (
    M1CPUOptimizer, M1BatchProcessor, get_m1_cpu_optimizer,
    initialize_m1_cpu_optimizer, parallel_map as parallel_map_cpu,
    parallel_dataframe_operation, parallel_monte_carlo_simulation,
    optimized_monte_carlo_worker
)

T = TypeVar('T')

@dataclass
class UtilityConfig:
    """Configuration for utility initialization."""
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_math_validation: bool = True
    enable_data_validation: bool = True
    enable_serialization: bool = True
    memory_limit_gb: float = 8.0
    max_workers: int = 4
    gpu_memory_threshold: float = 0.8
    log_level: str = 'INFO'

class Step05DependencyContainer:
    """
    Dependency injection container for Step05 utilities.
    
    This container manages the lifecycle and configuration of all utility dependencies
    used in Step05, providing a centralized way to access and configure utilities.
    """
    
    def __init__(self, config: Optional[UtilityConfig] = None):
        """Initialize the dependency container."""
        self.config = config or UtilityConfig()
        self.logger = get_logger('Step05DependencyContainer')
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialized = False
        
        # Register all utility factories
        self._register_factories()
        
        self.logger.info("🔧 Step05 Dependency Container initialized")
    
    def _register_factories(self):
        """Register factory functions for all utilities."""
        
        # Common operations utilities (stateless functions)
        self._factories['common_operations'] = lambda: {
            'datetime_ops': {
                'get_current_datetime': get_current_datetime,
                'get_today': get_today,
                'format_datetime': format_datetime,
                'parse_datetime': parse_datetime
            },
            'dataframe_ops': {
                'create_empty_dataframe': create_empty_dataframe,
                'safe_fillna': safe_fillna,
                'safe_rolling': safe_rolling,
                'safe_copy': safe_copy,
                'safe_deepcopy': safe_deepcopy
            },
            'math_ops': {
                'safe_mean': safe_mean,
                'safe_std': safe_std,
                'safe_float': safe_float,
                'safe_int': safe_int
            },
            'file_ops': {
                'ensure_directory': ensure_directory,
                'safe_file_exists': safe_file_exists,
                'safe_json_dump': safe_json_dump,
                'safe_json_load': safe_json_load,
                'safe_read_parquet': safe_read_parquet,
                'safe_to_parquet': safe_to_parquet
            },
            'async_ops': {
                'safe_sleep': safe_sleep,
                'safe_gather': safe_gather,
                'create_async_task': create_async_task
            },
            'list_ops': {
                'safe_append': safe_append,
                'safe_extend': safe_extend,
                'safe_dict_get': safe_dict_get,
                'safe_dict_items': safe_dict_items
            },
            'string_ops': {
                'safe_lower': safe_lower,
                'safe_upper': safe_upper,
                'safe_join': safe_join
            },
            'logging_ops': {
                'get_logger': get_logger,
                'setup_basic_logging': setup_basic_logging,
                'safe_log_metric': safe_log_metric,
                'safe_log_params': safe_log_params,
                'safe_log_artifact': safe_log_artifact
            },
            'validation_ops': {
                'validate_dataframe': validate_dataframe,
                'validate_numeric_range': validate_numeric_range,
                'validate_dataframe_schema': validate_dataframe_schema,
                'validate_data_quality': validate_data_quality
            },
            'optimization_ops': {
                'optimize_dataframe_dtypes': optimize_dataframe_dtypes,
                'timed_operation': timed_operation,
                'format_bytes': format_bytes,
                'chunked_iterable': chunked_iterable,
                'parallel_map': parallel_map
            },
            'specialized_ops': {
                'standardize_price_action_probabilities': standardize_price_action_probabilities,
                'suggest_float_uniform': suggest_float_uniform,
                'suggest_int_uniform': suggest_int_uniform,
                'safe_exception_handler': safe_exception_handler
            }
        }
        
        # Common utilities (stateless functions)
        self._factories['common_utilities'] = lambda: {
            'dataframe_operations': {
                'safe_dataframe_operation': safe_dataframe_operation,
                'validate_dataframe_columns': validate_dataframe_columns,
                'safe_convert_dtypes': safe_convert_dtypes,
                'safe_merge_dataframes': safe_merge_dataframes,
                'safe_groupby_operation': safe_groupby_operation,
                'safe_apply_function': safe_apply_function,
                'safe_drop_columns': safe_drop_columns,
                'safe_rename_columns': safe_rename_columns,
                'safe_filter_dataframe': safe_filter_dataframe
            },
            'data_analysis': {
                'calculate_data_quality_metrics': calculate_data_quality_metrics,
                'create_summary_statistics': create_summary_statistics,
                'get_dataframe_info': get_dataframe_info,
                'create_data_quality_report': create_data_quality_report
            },
            'timestamp_operations': {
                'validate_timestamp_column': validate_timestamp_column,
                'safe_timestamp_conversion': safe_timestamp_conversion
            }
        }
        
        # Math validation utilities (stateless functions)
        if self.config.enable_math_validation:
            self._factories['math_validation'] = lambda: {
                'safe_math_ops': {
                    'safe_divide': safe_divide,
                    'safe_log': safe_log,
                    'safe_sqrt': safe_sqrt,
                    'safe_power': safe_power
                },
                'validation_ops': {
                    'validate_finite': validate_finite,
                    'validate_positive': validate_positive,
                    'validate_range': validate_range
                },
                'financial_math': {
                    'safe_kelly_calculation': safe_kelly_calculation,
                    'safe_weighted_average': safe_weighted_average,
                    'safe_percentage_change': safe_percentage_change
                },
                'matrix_ops': {
                    'validate_correlation_matrix': validate_correlation_matrix,
                    'safe_matrix_inverse': safe_matrix_inverse
                },
                'decorators': {
                    'math_safe': math_safe
                },
                'exceptions': {
                    'MathValidationError': MathValidationError
                }
            }
        
        # Parquet utilities (singleton instance)
        self._factories['parquet_utils'] = lambda: {
            'parquet_utils': get_parquet_utils(),
            'parquet_utils_class': ParquetUtils
        }
        
        # Serialization utilities (stateless classes)
        if self.config.enable_serialization:
            self._factories['serialization_utils'] = lambda: {
                'serializers': {
                    'JSONSerializer': JSONSerializer,
                    'PickleSerializer': PickleSerializer,
                    'ParquetSerializer': ParquetSerializer,
                    'UniversalSerializer': UniversalSerializer
                },
                'convenience_functions': {
                    'save_json': save_json,
                    'load_json': load_json,
                    'save_pickle': save_pickle,
                    'load_pickle': load_pickle,
                    'save_parquet': save_parquet,
                    'load_parquet': load_parquet,
                    'save_data': save_data,
                    'load_data': load_data
                },
                'exceptions': {
                    'SerializationError': SerializationError
                }
            }
        
        # Data processing utilities (stateful classes)
        if self.config.enable_data_validation:
            self._factories['data_processing_utils'] = lambda: {
                'validators': {
                    'DataFrameValidator': DataFrameValidator,
                    'DataFrameCleaner': DataFrameCleaner,
                    'DataFrameTransformer': DataFrameTransformer
                },
                'convenience_functions': {
                    'validate_dataframe': validate_dataframe_advanced,
                    'clean_dataframe': clean_dataframe,
                    'transform_dataframe': transform_dataframe,
                    'get_dataframe_info': get_dataframe_info_advanced
                },
                'data_classes': {
                    'DataQualityLevel': DataQualityLevel,
                    'DataQualityIssue': DataQualityIssue,
                    'DataQualityReport': DataQualityReport
                }
            }
        
        # M1 GPU utilities (singleton instance)
        if self.config.enable_gpu_optimization:
            self._factories['m1_gpu_utils'] = lambda: {
                'gpu_manager': get_m1_gpu_manager(),
                'performance_optimizer': M1PerformanceOptimizer(get_m1_gpu_manager()),
                'convenience_functions': {
                    'm1_tensor_multiply': m1_tensor_multiply,
                    'm1_batch_process': m1_batch_process,
                    'm1_monte_carlo_simulate': m1_monte_carlo_simulate
                },
                'config_functions': {
                    'create_m1_optimized_config': create_m1_optimized_config,
                    'initialize_m1_gpu': initialize_m1_gpu
                }
            }
        
        # M1 Memory utilities (singleton instance)
        if self.config.enable_memory_optimization:
            self._factories['m1_memory_utils'] = lambda: {
                'memory_optimizer': get_m1_memory_optimizer(),
                'data_manager': M1DataManager(get_m1_memory_optimizer()),
                'convenience_functions': {
                    'create_memory_efficient_dataframe': create_memory_efficient_dataframe,
                    'memory_efficient_groupby': memory_efficient_groupby
                }
            }
        
        # M1 CPU utilities (singleton instance)
        if self.config.enable_cpu_optimization:
            self._factories['m1_cpu_utils'] = lambda: {
                'cpu_optimizer': get_m1_cpu_optimizer(),
                'batch_processor': M1BatchProcessor(get_m1_cpu_optimizer(), batch_size=1000),
                'convenience_functions': {
                    'parallel_map': parallel_map_cpu,
                    'parallel_dataframe_operation': parallel_dataframe_operation,
                    'parallel_monte_carlo_simulation': parallel_monte_carlo_simulation,
                    'optimized_monte_carlo_worker': optimized_monte_carlo_worker
                },
                'config_functions': {
                    'initialize_m1_cpu_optimizer': initialize_m1_cpu_optimizer
                }
            }
    
    def get_utility(self, utility_name: str, category: Optional[str] = None) -> Any:
        """
        Get a utility instance or function.
        
        Args:
            utility_name: Name of the utility to retrieve
            category: Optional category to narrow down the search
            
        Returns:
            The requested utility instance or function
        """
        if not self._initialized:
            self._initialize_all()
        
        if category:
            key = f"{category}.{utility_name}"
            if key in self._instances:
                return self._instances[key]
        
        # Search across all categories
        for cat_name, cat_utils in self._instances.items():
            if isinstance(cat_utils, dict):
                if utility_name in cat_utils:
                    return cat_utils[utility_name]
                # Search in nested dictionaries
                for sub_cat_name, sub_cat_utils in cat_utils.items():
                    if isinstance(sub_cat_utils, dict) and utility_name in sub_cat_utils:
                        return sub_cat_utils[utility_name]
        
        raise ValueError(f"Utility '{utility_name}' not found in container")
    
    def get_category(self, category_name: str) -> Dict[str, Any]:
        """
        Get all utilities in a specific category.
        
        Args:
            category_name: Name of the category
            
        Returns:
            Dictionary of utilities in the category
        """
        if not self._initialized:
            self._initialize_all()
        
        if category_name in self._instances:
            return self._instances[category_name]
        
        raise ValueError(f"Category '{category_name}' not found in container")
    
    def _initialize_all(self):
        """Initialize all utility instances."""
        if self._initialized:
            return
        
        self.logger.info("🔧 Initializing all utility dependencies...")
        
        for factory_name, factory_func in self._factories.items():
            try:
                self._instances[factory_name] = factory_func()
                self.logger.debug(f"✅ Initialized {factory_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {factory_name}: {e}")
                # Continue with other utilities even if one fails
                self._instances[factory_name] = {}
        
        self._initialized = True
        self.logger.info(f"✅ Initialized {len(self._instances)} utility categories")
    
    def get_all_utilities(self) -> Dict[str, Any]:
        """Get all initialized utilities."""
        if not self._initialized:
            self._initialize_all()
        return self._instances.copy()
    
    def get_utility_summary(self) -> Dict[str, Any]:
        """Get a summary of all available utilities."""
        if not self._initialized:
            self._initialize_all()
        
        summary = {}
        for category_name, category_utils in self._instances.items():
            if isinstance(category_utils, dict):
                summary[category_name] = {
                    'type': 'category',
                    'subcategories': list(category_utils.keys()),
                    'total_utilities': sum(
                        len(sub_cat) if isinstance(sub_cat, dict) else 1
                        for sub_cat in category_utils.values()
                    )
                }
            else:
                summary[category_name] = {
                    'type': 'instance',
                    'class': type(category_utils).__name__
                }
        
        return summary
    
    def configure_utility(self, utility_name: str, config: Dict[str, Any]) -> bool:
        """
        Configure a specific utility with custom settings.
        
        Args:
            utility_name: Name of the utility to configure
            config: Configuration dictionary
            
        Returns:
            True if configuration was successful
        """
        try:
            utility = self.get_utility(utility_name)
            
            # Apply configuration based on utility type
            if hasattr(utility, 'configure'):
                utility.configure(config)
            elif hasattr(utility, '__dict__'):
                for key, value in config.items():
                    if hasattr(utility, key):
                        setattr(utility, key, value)
            
            self.logger.info(f"✅ Configured utility '{utility_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to configure utility '{utility_name}': {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all utilities."""
        health_status = {
            'overall_status': 'healthy',
            'utilities_checked': 0,
            'utilities_healthy': 0,
            'utilities_failed': 0,
            'details': {}
        }
        
        if not self._initialized:
            self._initialize_all()
        
        for category_name, category_utils in self._instances.items():
            category_health = {
                'status': 'healthy',
                'utilities_checked': 0,
                'utilities_healthy': 0,
                'utilities_failed': 0,
                'details': {}
            }
            
            if isinstance(category_utils, dict):
                for sub_cat_name, sub_cat_utils in category_utils.items():
                    if isinstance(sub_cat_utils, dict):
                        for util_name, util_instance in sub_cat_utils.items():
                            try:
                                # Basic health check - try to access the utility
                                if callable(util_instance):
                                    # For functions, just check if they're callable
                                    category_health['utilities_healthy'] += 1
                                elif hasattr(util_instance, 'health_check'):
                                    # For classes with health check method
                                    if util_instance.health_check():
                                        category_health['utilities_healthy'] += 1
                                    else:
                                        category_health['utilities_failed'] += 1
                                else:
                                    # For other instances, just check if they exist
                                    category_health['utilities_healthy'] += 1
                                
                                category_health['utilities_checked'] += 1
                                
                            except Exception as e:
                                category_health['utilities_failed'] += 1
                                category_health['utilities_checked'] += 1
                                category_health['details'][util_name] = str(e)
                    else:
                        # Single utility in subcategory
                        try:
                            if callable(sub_cat_utils) or hasattr(sub_cat_utils, '__dict__'):
                                category_health['utilities_healthy'] += 1
                            else:
                                category_health['utilities_failed'] += 1
                            category_health['utilities_checked'] += 1
                        except Exception as e:
                            category_health['utilities_failed'] += 1
                            category_health['utilities_checked'] += 1
                            category_health['details'][sub_cat_name] = str(e)
            
            # Update category status
            if category_health['utilities_failed'] > 0:
                category_health['status'] = 'degraded' if category_health['utilities_healthy'] > 0 else 'failed'
            
            health_status['details'][category_name] = category_health
            health_status['utilities_checked'] += category_health['utilities_checked']
            health_status['utilities_healthy'] += category_health['utilities_healthy']
            health_status['utilities_failed'] += category_health['utilities_failed']
        
        # Update overall status
        if health_status['utilities_failed'] > 0:
            health_status['overall_status'] = 'degraded' if health_status['utilities_healthy'] > 0 else 'failed'
        
        return health_status


# Global container instance
_container: Optional[Step05DependencyContainer] = None

def get_step05_container(config: Optional[UtilityConfig] = None) -> Step05DependencyContainer:
    """Get the global Step05 dependency container instance."""
    global _container
    if _container is None:
        _container = Step05DependencyContainer(config)
    return _container

def initialize_step05_utilities(config: Optional[UtilityConfig] = None) -> Step05DependencyContainer:
    """Initialize Step05 utilities with dependency injection."""
    container = get_step05_container(config)
    container._initialize_all()
    return container

def get_utility(utility_name: str, category: Optional[str] = None) -> Any:
    """Convenience function to get a utility from the global container."""
    return get_step05_container().get_utility(utility_name, category)

def get_category(category_name: str) -> Dict[str, Any]:
    """Convenience function to get a category from the global container."""
    return get_step05_container().get_category(category_name)

# Export main classes and functions
__all__ = [
    'Step05DependencyContainer',
    'UtilityConfig',
    'get_step05_container',
    'initialize_step05_utilities',
    'get_utility',
    'get_category'
]