"""
Unified Utility Registry for Enhanced Mutual Use

This module provides a centralized registry and factory system for all utility modules,
enabling better integration, dependency management, and mutual use across the codebase.

Key Features:
- Centralized utility registration and discovery
- Dependency injection container
- Cross-utility integration patterns
- Performance optimization coordination
- Unified configuration management
- Health monitoring and diagnostics
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import asyncio
from enum import Enum
import weakref

# Import all utility modules for registration
from .common_operations import (
    get_current_datetime, safe_json_dump, safe_json_load, safe_file_exists,
    ensure_directory, safe_mean, safe_std, safe_float, safe_int,
    validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    get_logger, setup_basic_logging
)

from .common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)

from .math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, MathValidationError
)

from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

from .serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
    save_data, load_data, SerializationError
)

from .data_processing_utils import (
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
    validate_dataframe as validate_dataframe_advanced, clean_dataframe,
    transform_dataframe, get_dataframe_info as get_dataframe_info_advanced
)

from .hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager, initialize_m1_gpu

from .hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer

from .hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer

logger = logging.getLogger(__name__)

T = TypeVar('T')

class UtilityCategory(Enum):
    """Categories for utility classification."""
    DATA_OPERATIONS = "data_operations"
    MATHEMATICAL = "mathematical"
    FILE_IO = "file_io"
    SERIALIZATION = "serialization"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    GPU = "gpu"
    CPU = "cpu"
    ML_COMMON = "ml_common"

@dataclass
class UtilityMetadata:
    """Metadata for registered utilities."""
    name: str
    category: UtilityCategory
    version: str
    dependencies: List[str] = field(default_factory=list)
    performance_impact: str = "low"  # low, medium, high
    memory_usage: str = "low"  # low, medium, high
    thread_safe: bool = True
    async_safe: bool = False
    description: str = ""
    last_used: Optional[float] = None
    usage_count: int = 0

class UtilityRegistry:
    """Centralized registry for all utility modules."""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, UtilityMetadata] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._health_status: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with core utilities
        self._register_core_utilities()
        self._register_ml_common_utilities()
    
    def _register_core_utilities(self):
        """Register core utility functions and classes."""
        
        # Data Operations
        self.register("safe_json_dump", safe_json_dump, UtilityCategory.FILE_IO,
                     description="Safely dump data to JSON with error handling")
        self.register("safe_json_load", safe_json_load, UtilityCategory.FILE_IO,
                     description="Safely load data from JSON with error handling")
        self.register("safe_file_exists", safe_file_exists, UtilityCategory.FILE_IO,
                     description="Safely check file existence")
        self.register("ensure_directory", ensure_directory, UtilityCategory.FILE_IO,
                     description="Ensure directory exists, creating if necessary")
        
        # Mathematical Operations
        self.register("safe_divide", safe_divide, UtilityCategory.MATHEMATICAL,
                     description="Safe division with zero protection")
        self.register("safe_log", safe_log, UtilityCategory.MATHEMATICAL,
                     description="Safe logarithm calculation")
        self.register("safe_sqrt", safe_sqrt, UtilityCategory.MATHEMATICAL,
                     description="Safe square root calculation")
        self.register("safe_mean", safe_mean, UtilityCategory.MATHEMATICAL,
                     description="Safe mean calculation with error handling")
        self.register("safe_std", safe_std, UtilityCategory.MATHEMATICAL,
                     description="Safe standard deviation calculation")
        
        # DataFrame Operations
        self.register("safe_dataframe_operation", safe_dataframe_operation, UtilityCategory.DATA_OPERATIONS,
                     description="Safely perform operations on DataFrames")
        self.register("validate_dataframe_columns", validate_dataframe_columns, UtilityCategory.VALIDATION,
                     description="Validate DataFrame column requirements")
        self.register("calculate_data_quality_metrics", calculate_data_quality_metrics, UtilityCategory.VALIDATION,
                     description="Calculate comprehensive data quality metrics")
        
        # Serialization
        self.register("JSONSerializer", JSONSerializer, UtilityCategory.SERIALIZATION,
                     description="JSON serialization utilities")
        self.register("ParquetSerializer", ParquetSerializer, UtilityCategory.SERIALIZATION,
                     description="Parquet serialization utilities")
        self.register("UniversalSerializer", UniversalSerializer, UtilityCategory.SERIALIZATION,
                     description="Universal serialization with auto-format detection")
        
        # Performance Utilities
        self.register("M1GPUManager", M1GPUManager, UtilityCategory.GPU,
                     description="M1 GPU optimization manager")
        self.register("M1MemoryOptimizer", M1MemoryOptimizer, UtilityCategory.MEMORY,
                     description="M1 memory optimization utilities")
        self.register("M1CPUOptimizer", M1CPUOptimizer, UtilityCategory.CPU,
                     description="M1 CPU optimization utilities")
        
        # Advanced Data Processing
        self.register("DataFrameValidator", DataFrameValidator, UtilityCategory.VALIDATION,
                     description="Comprehensive DataFrame validation")
        self.register("DataFrameCleaner", DataFrameCleaner, UtilityCategory.DATA_OPERATIONS,
                     description="DataFrame cleaning and preprocessing")
        self.register("DataFrameTransformer", DataFrameTransformer, UtilityCategory.DATA_OPERATIONS,
                     description="DataFrame transformation utilities")
        
        # Parquet Utilities
        self.register("ParquetUtils", ParquetUtils, UtilityCategory.FILE_IO,
                     description="Advanced parquet file operations")
    
    def register(self, name: str, utility: Any, category: UtilityCategory, 
                dependencies: Optional[List[str]] = None, **metadata) -> None:
        """Register a utility with metadata."""
        with self._lock:
            self._registry[name] = utility
            
            # Create metadata
            meta = UtilityMetadata(
                name=name,
                category=category,
                version=metadata.get('version', '1.0.0'),
                dependencies=dependencies or [],
                performance_impact=metadata.get('performance_impact', 'low'),
                memory_usage=metadata.get('memory_usage', 'low'),
                thread_safe=metadata.get('thread_safe', True),
                async_safe=metadata.get('async_safe', False),
                description=metadata.get('description', '')
            )
            self._metadata[name] = meta
            
            logger.debug(f"✅ Registered utility: {name} ({category.value})")
    
    def get(self, name: str, create_instance: bool = False) -> Any:
        """Get a registered utility."""
        with self._lock:
            if name not in self._registry:
                raise KeyError(f"Utility '{name}' not found in registry")
            
            utility = self._registry[name]
            
            # Update usage statistics
            if name in self._metadata:
                self._metadata[name].usage_count += 1
                self._metadata[name].last_used = time.time()
            
            # Create instance if requested and it's a class
            if create_instance and isinstance(utility, type):
                if name not in self._instances:
                    self._instances[name] = utility()
                return self._instances[name]
            
            return utility
    
    def get_by_category(self, category: UtilityCategory) -> Dict[str, Any]:
        """Get all utilities in a category."""
        with self._lock:
            return {
                name: utility for name, utility in self._registry.items()
                if self._metadata.get(name, {}).category == category
            }
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a utility."""
        with self._lock:
            return self._metadata.get(name, {}).dependencies or []
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all utilities."""
        with self._lock:
            return {
                name: {
                    'usage_count': meta.usage_count,
                    'last_used': meta.last_used,
                    'category': meta.category.value,
                    'performance_impact': meta.performance_impact,
                    'memory_usage': meta.memory_usage
                }
                for name, meta in self._metadata.items()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all utilities."""
        with self._lock:
            health_report = {
                'total_utilities': len(self._registry),
                'categories': {},
                'health_status': 'healthy',
                'issues': []
            }
            
            # Check each category
            for category in UtilityCategory:
                category_utils = self.get_by_category(category)
                health_report['categories'][category.value] = {
                    'count': len(category_utils),
                    'utilities': list(category_utils.keys())
                }
            
            # Check for unused utilities
            unused_utilities = [
                name for name, meta in self._metadata.items()
                if meta.usage_count == 0
            ]
            
            if unused_utilities:
                health_report['issues'].append(f"Unused utilities: {unused_utilities}")
                health_report['health_status'] = 'warning'
            
            return health_report

    def _register_ml_common_utilities(self):
        """Register ML Common utilities."""
        try:
            # Import ML Common utilities
            from src.utils.ml_common import (
                UnifiedCrossValidator, UnifiedCVResult, perform_cross_validation,
                EnhancedValidator, EnhancedValidationConfig,
                get_enhanced_validator, validate_model_comprehensively
            )

            # Register cross-validation utilities
            self.register(
                "unified_cross_validator",
                UnifiedCrossValidator,
                UtilityCategory.ML_COMMON,
                description="Unified cross-validation system",
                performance_impact="medium",
                memory_usage="medium"
            )

            self.register(
                "perform_cross_validation",
                perform_cross_validation,
                UtilityCategory.ML_COMMON,
                description="Perform cross-validation with multiple strategies",
                performance_impact="medium",
                memory_usage="medium"
            )

            # Register enhanced validation utilities
            self.register(
                "enhanced_validator",
                EnhancedValidator,
                UtilityCategory.ML_COMMON,
                description="Comprehensive model validation system",
                performance_impact="high",
                memory_usage="high"
            )

            self.register(
                "get_enhanced_validator",
                get_enhanced_validator,
                UtilityCategory.ML_COMMON,
                description="Get enhanced validator instance",
                performance_impact="low",
                memory_usage="low"
            )

            logger.info("✅ ML Common utilities registered in utility registry")

        except Exception as e:
            logger.warning(f"⚠️ Failed to register ML Common utilities: {e}")

class UtilityFactory:
    """Factory for creating utility instances with dependency injection."""
    
    def __init__(self, registry: UtilityRegistry):
        self.registry = registry
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def create(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a utility instance with configuration."""
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            
            utility_class = self.registry.get(name)
            
            if not isinstance(utility_class, type):
                return utility_class
            
            # Resolve dependencies
            dependencies = self.registry.get_dependencies(name)
            resolved_deps = {}
            
            for dep_name in dependencies:
                resolved_deps[dep_name] = self.create(dep_name, config)
            
            # Create instance
            try:
                if config:
                    instance = utility_class(config, **resolved_deps)
                else:
                    instance = utility_class(**resolved_deps)
                
                self._instances[name] = instance
                logger.debug(f"✅ Created instance: {name}")
                return instance
                
            except Exception as e:
                logger.error(f"❌ Failed to create instance {name}: {e}")
                raise
    
    def get_or_create(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get existing instance or create new one."""
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            return self.create(name, config)

class CrossUtilityIntegrator:
    """Integrator for cross-utility operations and optimizations."""
    
    def __init__(self, registry: UtilityRegistry, factory: UtilityFactory):
        self.registry = registry
        self.factory = factory
        self._integration_cache: Dict[str, Any] = {}
    
    def create_data_pipeline(self, operations: List[Dict[str, Any]]) -> Callable:
        """Create an optimized data processing pipeline."""
        def pipeline(data: Any) -> Any:
            result = data
            
            for operation in operations:
                op_name = operation['name']
                op_params = operation.get('params', {})
                
                # Get utility
                utility = self.factory.get_or_create(op_name)
                
                # Apply operation
                if callable(utility):
                    result = utility(result, **op_params)
                else:
                    # Handle class-based utilities
                    method_name = operation.get('method', 'process')
                    method = getattr(utility, method_name, None)
                    if method:
                        result = method(result, **op_params)
            
            return result
        
        return pipeline
    
    def optimize_memory_usage(self, data_size_mb: float) -> Dict[str, Any]:
        """Optimize memory usage across utilities."""
        memory_optimizer = self.factory.get_or_create("M1MemoryOptimizer")
        
        # Get memory-efficient utilities
        memory_efficient_utils = [
            name for name, meta in self.registry._metadata.items()
            if meta.memory_usage == "low"
        ]
        
        return {
            'recommended_utilities': memory_efficient_utils,
            'chunk_size': memory_optimizer.calculate_optimal_chunk_size(
                (int(data_size_mb * 1024 * 1024 / 8),), "general"
            ),
            'should_chunk': memory_optimizer.should_chunk_data(data_size_mb, "general")
        }
    
    def optimize_performance(self, operation_type: str) -> Dict[str, Any]:
        """Optimize performance across utilities."""
        cpu_optimizer = self.factory.get_or_create("M1CPUOptimizer")
        gpu_manager = self.factory.get_or_create("M1GPUManager")
        
        return {
            'optimal_workers': cpu_optimizer.get_optimal_workers_for_task(operation_type),
            'use_gpu': gpu_manager.should_use_gpu(10000, operation_type),
            'performance_utilities': [
                name for name, meta in self.registry._metadata.items()
                if meta.performance_impact == "high"
            ]
        }

# Global registry instance
_registry = None
_factory = None
_integrator = None

def get_utility_registry() -> UtilityRegistry:
    """Get global utility registry instance."""
    global _registry
    if _registry is None:
        _registry = UtilityRegistry()
    return _registry

def get_utility_factory() -> UtilityFactory:
    """Get global utility factory instance."""
    global _factory
    if _factory is None:
        _factory = UtilityFactory(get_utility_registry())
    return _factory

def get_cross_utility_integrator() -> CrossUtilityIntegrator:
    """Get global cross-utility integrator instance."""
    global _integrator
    if _integrator is None:
        _integrator = CrossUtilityIntegrator(get_utility_registry(), get_utility_factory())
    return _integrator

# Convenience functions
def get_utility(name: str, create_instance: bool = False) -> Any:
    """Get a utility from the registry."""
    return get_utility_registry().get(name, create_instance)

def create_utility_instance(name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create a utility instance with dependency injection."""
    return get_utility_factory().create(name, config)

def get_utilities_by_category(category: UtilityCategory) -> Dict[str, Any]:
    """Get utilities by category."""
    return get_utility_registry().get_by_category(category)

def create_optimized_pipeline(operations: List[Dict[str, Any]]) -> Callable:
    """Create an optimized data processing pipeline."""
    return get_cross_utility_integrator().create_data_pipeline(operations)

def optimize_for_data_size(data_size_mb: float) -> Dict[str, Any]:
    """Get optimization recommendations for data size."""
    return get_cross_utility_integrator().optimize_memory_usage(data_size_mb)

def optimize_for_operation(operation_type: str) -> Dict[str, Any]:
    """Get optimization recommendations for operation type."""
    return get_cross_utility_integrator().optimize_performance(operation_type)

# Health monitoring
def get_utility_health_report() -> Dict[str, Any]:
    """Get comprehensive health report for all utilities."""
    return get_utility_registry().health_check()

def get_utility_usage_stats() -> Dict[str, Dict[str, Any]]:
    """Get usage statistics for all utilities."""
    def _register_ml_common_utilities(self):
        """Register ML Common utilities."""
        try:
            # Import ML Common utilities
            from src.utils.ml_common.validation import (
                UnifiedCrossValidator, UnifiedCVResult, perform_cross_validation,
                EnhancedValidator, EnhancedValidationConfig,
                get_enhanced_validator, validate_model_comprehensively
            )

            # Register cross-validation utilities
            self.register(
                "unified_cross_validator",
                UnifiedCrossValidator,
                UtilityCategory.ML_COMMON,
                description="Unified cross-validation system",
                performance_impact="medium",
                memory_usage="medium"
            )

            self.register(
                "perform_cross_validation",
                perform_cross_validation,
                UtilityCategory.ML_COMMON,
                description="Perform cross-validation with multiple strategies",
                performance_impact="medium",
                memory_usage="medium"
            )

            # Register enhanced validation utilities
            self.register(
                "enhanced_validator",
                EnhancedValidator,
                UtilityCategory.ML_COMMON,
                description="Comprehensive model validation system",
                performance_impact="high",
                memory_usage="high"
            )

            self.register(
                "get_enhanced_validator",
                get_enhanced_validator,
                UtilityCategory.ML_COMMON,
                description="Get enhanced validator instance",
                performance_impact="low",
                memory_usage="low"
            )

            logger.info("✅ ML Common utilities registered in utility registry")

        except Exception as e:
            logger.warning(f"⚠️ Failed to register ML Common utilities: {e}")

    return get_utility_registry().get_usage_stats()

__all__ = [
    'UtilityRegistry', 'UtilityFactory', 'CrossUtilityIntegrator',
    'UtilityCategory', 'UtilityMetadata',
    'get_utility_registry', 'get_utility_factory', 'get_cross_utility_integrator',
    'get_utility', 'create_utility_instance', 'get_utilities_by_category',
    'create_optimized_pipeline', 'optimize_for_data_size', 'optimize_for_operation',
    'get_utility_health_report', 'get_utility_usage_stats'
]
