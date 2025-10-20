from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)

from __future__ import annotations

"""
Memory Integration Module for ML Common Utilities

This module provides comprehensive integration of automatic memory skimming
with all ML common utilities, ensuring optimal memory management across
all machine learning operations.

Key Features:
- Automatic memory skimming for all ML operations
- Memory-aware decorators for ML functions
- Context managers for memory-intensive operations
- Integration with existing ML utilities
- Memory monitoring and reporting
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
from contextlib import contextmanager

# Import memory skimming utilities

# Define memory management functions with actual implementation
def auto_skim_memory(memory_mb: float, operation_type: str) -> Dict[str, Any]:
    """Auto memory skimming for optimal memory usage."""
    logger = logging.getLogger(__name__)
    logger.debug(f"🔧 Auto-skimming {memory_mb:.2f} MB for {operation_type}")

    try:
        # Get the memory optimizer
        memory_optimizer = get_integrated_hardware_manager()

        # Get initial memory stats
        initial_stats = memory_optimizer.get_memory_stats()
        initial_memory_mb = initial_stats.get('used_memory', 0) / (1024 * 1024)

        # Determine optimization level based on requested memory
        if memory_mb > 2000:  # > 2GB
            optimization_result = memory_optimizer.optimize_memory_usage(aggressive=True)
        elif memory_mb > 1000:  # > 1GB
            optimization_result = memory_optimizer.optimize_memory_usage(aggressive=False)
        else:
            # Light optimization for smaller requests
            memory_optimizer._light_memory_cleanup()
            optimization_result = {'success': True, 'memory_saved_mb': 0}

        # Get final memory stats
        final_stats = memory_optimizer.get_memory_stats()
        final_memory_mb = final_stats.get('used_memory', 0) / (1024 * 1024)

        # Calculate actual memory freed
        memory_freed_mb = initial_memory_mb - final_memory_mb

        # If we didn't free enough, try aggressive cleanup
        if memory_freed_mb < memory_mb * 0.1:  # Less than 10% of requested
            logger.warning(f"⚠️ Insufficient memory freed ({memory_freed_mb:.1f} MB), trying aggressive cleanup")
            memory_optimizer._aggressive_memory_cleanup()

            # Recalculate after aggressive cleanup
            final_stats = memory_optimizer.get_memory_stats()
            final_memory_mb = final_stats.get('used_memory', 0) / (1024 * 1024)
            memory_freed_mb = initial_memory_mb - final_memory_mb

        return {
            'memory_freed_mb': max(memory_freed_mb, 0),
            'operation_type': operation_type,
            'success': True,
            'skimming_performed': True,
            'initial_memory_mb': initial_memory_mb,
            'final_memory_mb': final_memory_mb,
            'optimization_result': optimization_result
        }

    except Exception as e:
        logger.error(f"❌ Memory skimming failed: {e}")
        return {
            'memory_freed_mb': 0,
            'operation_type': operation_type,
            'success': False,
            'skimming_performed': False,
            'error': str(e)
        }

def smart_memory_allocation(memory_mb: float, operation_type: str) -> Dict[str, Any]:
    """Smart memory allocation based on operation type."""
    logger = logging.getLogger(__name__)
    logger.debug(f"🔧 Smart allocating {memory_mb:.2f} MB for {operation_type}")

    try:
        # Get the memory optimizer
        memory_optimizer = get_integrated_hardware_manager()

        # Get current memory stats
        current_stats = memory_optimizer.get_memory_stats()
        current_memory_mb = current_stats.get('used_memory', 0) / (1024 * 1024)
        total_memory_mb = current_stats.get('total_memory', 0) / (1024 * 1024)
        available_memory_mb = current_stats.get('available_memory', 0) / (1024 * 1024)

        # Check if we have enough available memory
        if available_memory_mb < memory_mb:
            logger.warning(f"⚠️ Insufficient available memory: {available_memory_mb:.1f} MB < {memory_mb:.1f} MB")

            # Try to free up memory
            logger.info("🧹 Attempting to free up memory for allocation")
            skim_result = auto_skim_memory(memory_mb, operation_type)

            # Recheck available memory after skimming
            updated_stats = memory_optimizer.get_memory_stats()
            updated_available_mb = updated_stats.get('available_memory', 0) / (1024 * 1024)

            if updated_available_mb < memory_mb:
                logger.error(f"❌ Still insufficient memory after skimming: {updated_available_mb:.1f} MB < {memory_mb:.1f} MB")
                return {
                    'allocated_mb': 0,
                    'operation_type': operation_type,
                    'optimization_applied': True,
                    'allocation_successful': False,
                    'error': 'Insufficient memory after optimization',
                    'available_memory_mb': updated_available_mb,
                    'requested_memory_mb': memory_mb,
                    'skim_result': skim_result
                }

        # Check memory pressure and apply appropriate optimizations
        memory_pressure = current_stats.get('memory_percent', 0) / 100.0

        if memory_pressure > 0.85:  # High memory pressure
            logger.warning(f"⚠️ High memory pressure: {memory_pressure:.1%}")
            memory_optimizer._aggressive_memory_cleanup()
        elif memory_pressure > 0.75:  # Medium memory pressure
            logger.info(f"🧠 Medium memory pressure: {memory_pressure:.1%}")
            memory_optimizer._moderate_memory_cleanup()
        elif memory_pressure > 0.6:  # Low memory pressure
            memory_optimizer._light_memory_cleanup()

        # Get final memory stats
        final_stats = memory_optimizer.get_memory_stats()
        final_available_mb = final_stats.get('available_memory', 0) / (1024 * 1024)

        # Determine if allocation was successful
        allocation_successful = final_available_mb >= memory_mb

        return {
            'allocated_mb': memory_mb if allocation_successful else 0,
            'operation_type': operation_type,
            'optimization_applied': True,
            'allocation_successful': allocation_successful,
            'available_memory_mb': final_available_mb,
            'requested_memory_mb': memory_mb,
            'memory_pressure': memory_pressure,
            'total_memory_mb': total_memory_mb,
            'current_memory_mb': current_memory_mb
        }

    except Exception as e:
        logger.error(f"❌ Smart memory allocation failed: {e}")
        return {
            'allocated_mb': 0,
            'operation_type': operation_type,
            'optimization_applied': False,
            'allocation_successful': False,
            'error': str(e)
        }

def memory_skim_decorator(operation_type: str):
    """Decorator for memory skimming operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            logger.debug(f"🔧 Memory skim decorator for {operation_type}")

            # Pre-operation memory management
            try:
                memory_optimizer = get_integrated_hardware_manager()
                initial_stats = memory_optimizer.get_memory_stats()
                initial_memory_mb = initial_stats.get('used_memory', 0) / (1024 * 1024)

                # Estimate memory requirements for the operation
                estimated_memory = 100.0  # Default 100MB estimate
                if hasattr(func, '__name__'):
                    if 'train' in func.__name__.lower() or 'fit' in func.__name__.lower():
                        estimated_memory = 500.0
                    elif 'predict' in func.__name__.lower() or 'inference' in func.__name__.lower():
                        estimated_memory = 200.0
                    elif 'optimize' in func.__name__.lower() or 'hyperparameter' in func.__name__.lower():
                        estimated_memory = 1000.0

                # Pre-allocate memory if needed
                allocation_result = smart_memory_allocation(estimated_memory, operation_type)
                if not allocation_result['allocation_successful']:
                    logger.warning(f"⚠️ Memory allocation failed for {func.__name__}")

            except Exception as e:
                logger.warning(f"⚠️ Pre-operation memory management failed: {e}")

            # Execute the function
            try:
                result = func(*args, **kwargs)

                # Post-operation cleanup
                try:
                    auto_skim_memory(50.0, operation_type)  # Clean up 50MB after operation
                except Exception as e:
                    logger.debug(f"Post-operation cleanup failed: {e}")

                return result

            except MemoryError as e:
                logger.error(f"❌ Memory error in {func.__name__}: {e}")
                # Try emergency cleanup
                try:
                    memory_optimizer = get_integrated_hardware_manager()
                    memory_optimizer._aggressive_memory_cleanup()
                except Exception as cleanup_error:
                    logger.error(f"❌ Emergency cleanup failed: {cleanup_error}")
                raise
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator

def auto_memory_skim_decorator(operation_type: str):
    """Auto memory skim decorator."""
    return memory_skim_decorator(operation_type)

@contextmanager
def auto_memory_skim_context(operation_type: str):
    """Context manager for auto memory skimming."""
    logger = logging.getLogger(__name__)
    logger.debug(f"🔧 Entering memory skim context for {operation_type}")

    # Pre-context memory management
    initial_stats = None
    try:
        memory_optimizer = get_integrated_hardware_manager()
        initial_stats = memory_optimizer.get_memory_stats()
        initial_memory_mb = initial_stats.get('used_memory', 0) / (1024 * 1024)

        # Pre-allocate memory based on operation type
        estimated_memory = {
            'hyperparameter_optimization': 1000.0,
            'cross_validation': 500.0,
            'model_training': 800.0,
            'feature_engineering': 300.0,
            'data_preprocessing': 200.0,
            'model_inference': 100.0,
            'general': 50.0
        }.get(operation_type, 100.0)

        allocation_result = smart_memory_allocation(estimated_memory, operation_type)
        if not allocation_result['allocation_successful']:
            logger.warning(f"⚠️ Memory allocation failed for {operation_type}")

    except Exception as e:
        logger.warning(f"⚠️ Pre-context memory management failed: {e}")

    try:
        yield
    finally:
        # Post-context cleanup
        try:
            auto_skim_memory(100.0, operation_type)  # Clean up 100MB after operation

            # Log memory usage if we have initial stats
            if initial_stats:
                final_stats = memory_optimizer.get_memory_stats()
                final_memory_mb = final_stats.get('used_memory', 0) / (1024 * 1024)
                memory_change = final_memory_mb - initial_memory_mb

                if memory_change > 50:  # Log if memory increased by more than 50MB
                    logger.info(f"🧠 Memory context '{operation_type}' completed: {memory_change:+.1f} MB")
                else:
                    logger.debug(f"🧠 Memory context '{operation_type}' completed: {memory_change:+.1f} MB")

        except Exception as e:
            logger.debug(f"Post-context cleanup failed: {e}")

        logger.debug(f"🔧 Exiting memory skim context for {operation_type}")

@contextmanager
def smart_memory_context(operation_type: str):
    """Context manager for smart memory allocation."""
    logger = logging.getLogger(__name__)
    logger.debug(f"🔧 Entering smart memory context for {operation_type}")

    # Pre-context smart memory allocation
    allocation_result = None
    try:
        # Estimate memory requirements
        estimated_memory = {
            'hyperparameter_optimization': 2000.0,
            'cross_validation': 1000.0,
            'model_training': 1500.0,
            'feature_engineering': 500.0,
            'data_preprocessing': 300.0,
            'model_inference': 200.0,
            'general': 100.0
        }.get(operation_type, 200.0)

        allocation_result = smart_memory_allocation(estimated_memory, operation_type)
        if not allocation_result['allocation_successful']:
            logger.warning(f"⚠️ Smart memory allocation failed for {operation_type}")

    except Exception as e:
        logger.warning(f"⚠️ Pre-context smart memory allocation failed: {e}")

    try:
        yield allocation_result
    finally:
        # Post-context cleanup and reporting
        try:
            if allocation_result and allocation_result.get('allocation_successful'):
                # Clean up allocated memory
                auto_skim_memory(estimated_memory * 0.5, operation_type)  # Clean up 50% of allocated memory

            # Log final memory stats
            memory_optimizer = get_integrated_hardware_manager()
            final_stats = memory_optimizer.get_memory_stats()
            final_memory_mb = final_stats.get('used_memory', 0) / (1024 * 1024)
            memory_pressure = final_stats.get('memory_percent', 0)

            logger.debug(f"🧠 Smart memory context '{operation_type}' completed: {final_memory_mb:.1f} MB used, {memory_pressure:.1f}% pressure")

        except Exception as e:
            logger.debug(f"Post-context smart memory cleanup failed: {e}")

        logger.debug(f"🔧 Exiting smart memory context for {operation_type}")

logger = logging.getLogger(__name__)

class MLMemoryManager:
    """Memory manager specifically designed for ML operations."""

    def __init__(self):
        self.logger = logger.getChild('MLMemoryManager')
        self.logger.info("🚀 Initializing MLMemoryManager...")
        start_time = time.time()

        self.memory_optimizer = get_integrated_hardware_manager()
        self.logger.debug("✅ Memory optimizer initialized")

        self.operation_memory_usage = {}
        self.logger.debug("✅ Operation memory usage tracking initialized")

        init_time = time.time() - start_time
        self.logger.info(f"✅ MLMemoryManager initialized in {init_time:.3f}s")

    def estimate_ml_memory_requirements(
        self,
        operation_type: str,
        data_shape: Optional[tuple] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        n_trials: Optional[int] = None,
        cv_folds: Optional[int] = None,
        **kwargs
    ) -> float:
        """Estimate memory requirements for ML operations."""

        # Base memory requirements by operation type
        base_requirements = {
            'hyperparameter_optimization': 2000,  # MB
            'cross_validation': 1500,
            'model_training': 1000,
            'feature_engineering': 800,
            'data_preprocessing': 600,
            'model_inference': 400,
            'lookahead_validation': 500,
            'temporal_validation': 300,
            'general': 200
        }

        base_mb = base_requirements.get(operation_type, base_requirements['general'])

        # Adjust based on data size
        if data_shape:
            data_elements = np.prod(data_shape)
            data_memory_mb = data_elements * 8 / (1024**2)  # Assume float64
            base_mb += data_memory_mb

        # Adjust based on samples and features
        if n_samples and n_features:
            matrix_memory_mb = n_samples * n_features * 8 / (1024**2)
            base_mb += matrix_memory_mb

        # Adjust based on trials and CV folds
        if n_trials:
            base_mb += n_trials * 100  # 100MB per trial
        if cv_folds:
            base_mb += cv_folds * 200  # 200MB per CV fold

        # Cap the estimation to reasonable limits
        estimated_mb = min(base_mb, 8000)  # Max 8GB

        self.logger.debug(f"📏 Estimated memory for {operation_type}: {estimated_mb:.1f}MB")
        return estimated_mb

    def auto_skim_for_ml_operation(
        self,
        operation_type: str,
        data_shape: Optional[tuple] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Automatically skim memory for ML operations."""

        estimated_memory_mb = self.estimate_ml_memory_requirements(
            operation_type, data_shape, **kwargs
        )

        return auto_skim_memory(estimated_memory_mb, operation_type)

    def smart_allocate_for_ml_operation(
        self,
        operation_type: str,
        data_shape: Optional[tuple] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Smart memory allocation for ML operations."""

        estimated_memory_mb = self.estimate_ml_memory_requirements(
            operation_type, data_shape, **kwargs
        )

        return smart_memory_allocation(estimated_memory_mb, operation_type)

# Global ML memory manager instance
_ml_memory_manager = None

def get_ml_memory_manager() -> MLMemoryManager:
    """Get global ML memory manager instance."""
    global _ml_memory_manager
    if _ml_memory_manager is None:
        _ml_memory_manager = MLMemoryManager()
    return _ml_memory_manager

# ML-specific decorators
def ml_memory_skim_decorator(operation_type: str):
    """Decorator for automatic memory skimming in ML operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_ml_memory_manager()

            # Estimate memory requirements
            estimated_memory_mb = manager.estimate_ml_memory_requirements(
                operation_type, **kwargs
            )

            # Perform smart memory allocation
            allocation_results = smart_memory_allocation(estimated_memory_mb, operation_type)

            if not allocation_results['allocation_successful']:
                logger.warning(f"⚠️ Insufficient memory for {func.__name__} even after skimming")

            # Execute the function
            try:
                result = func(*args, **kwargs)
                return result
            except MemoryError as e:
                logger.error(f"❌ Memory error in {func.__name__}: {e}")
                # Try emergency cleanup
                logger.info("🧹 Attempting emergency memory cleanup")
                manager.memory_optimizer._aggressive_memory_cleanup()
                raise
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator

def ml_auto_memory_skim_decorator():
    """Decorator that automatically determines memory requirements for ML operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_ml_memory_manager()

            # Try to determine operation type from function name
            operation_type = _infer_operation_type(func.__name__)

            # Estimate memory requirements
            estimated_memory_mb = manager.estimate_ml_memory_requirements(
                operation_type, **kwargs
            )

            # Perform smart memory allocation
            allocation_results = smart_memory_allocation(estimated_memory_mb, operation_type)

            if not allocation_results['allocation_successful']:
                logger.warning(f"⚠️ Insufficient memory for {func.__name__} even after skimming")

            # Execute the function
            try:
                result = func(*args, **kwargs)
                return result
            except MemoryError as e:
                logger.error(f"❌ Memory error in {func.__name__}: {e}")
                # Try emergency cleanup
                logger.info("🧹 Attempting emergency memory cleanup")
                manager.memory_optimizer._aggressive_memory_cleanup()
                raise
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator

def _infer_operation_type(function_name: str) -> str:
    """Infer operation type from function name."""
    function_name_lower = function_name.lower()

    if any(keyword in function_name_lower for keyword in ['optimize', 'hyperparameter', 'hpo']):
        return 'hyperparameter_optimization'
    elif any(keyword in function_name_lower for keyword in ['cross_validation', 'cv', 'validation']):
        return 'cross_validation'
    elif any(keyword in function_name_lower for keyword in ['train', 'fit', 'training']):
        return 'model_training'
    elif any(keyword in function_name_lower for keyword in ['feature', 'engineering', 'transform']):
        return 'feature_engineering'
    elif any(keyword in function_name_lower for keyword in ['preprocess', 'clean', 'process']):
        return 'data_preprocessing'
    elif any(keyword in function_name_lower for keyword in ['predict', 'inference', 'score']):
        return 'model_inference'
    elif any(keyword in function_name_lower for keyword in ['lookahead', 'bias', 'temporal']):
        return 'lookahead_validation'
    else:
        return 'general'

# ML-specific context managers
@contextmanager
def ml_memory_context(operation_type: str, **kwargs):
    """Context manager for ML operations with automatic memory management."""
    manager = get_ml_memory_manager()

    # Perform smart memory allocation
    allocation_results = manager.smart_allocate_for_ml_operation(operation_type, **kwargs)

    try:
        yield allocation_results
    finally:
        # Optional cleanup after operation
        if allocation_results['skimming_performed']:
            logger.debug(f"🧹 ML memory context completed for {operation_type}")

@contextmanager
def ml_auto_memory_context(**kwargs):
    """Context manager with automatic memory estimation for ML operations."""
    manager = get_ml_memory_manager()

    # Estimate memory requirements
    estimated_memory_mb = manager.estimate_ml_memory_requirements('general', **kwargs)

    # Perform smart memory allocation
    allocation_results = smart_memory_allocation(estimated_memory_mb, 'general')

    try:
        yield allocation_results
    finally:
        # Optional cleanup after operation
        if allocation_results['skimming_performed']:
            logger.debug("🧹 ML auto memory context completed")

# Integration functions for existing ML utilities
def integrate_memory_skimming_with_hpo():
    """Integrate memory skimming with hyperparameter optimization utilities."""
    try:
        from .hpo_utils import HyperparameterOptimization
        logger.info("✅ Successfully imported HPO utilities")
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with HPO utilities: {e}")
        return False

    # Initialize variables to avoid scoping issues
    original_multi_objective = None
    original_early_stopping = None

    try:
        # Store original methods before enhancement
        if hasattr(HyperparameterOptimization, 'multi_objective_optimization'):
            original_multi_objective = HyperparameterOptimization.multi_objective_optimization

        if hasattr(HyperparameterOptimization, 'early_stopping_optimization'):
            original_early_stopping = HyperparameterOptimization.early_stopping_optimization

        if original_multi_objective is not None:
            def enhanced_multi_objective_optimization(self, *args, **kwargs):
                try:
                    manager = get_ml_memory_manager()
                    estimated_memory_mb = manager.estimate_ml_memory_requirements(
                        'hyperparameter_optimization', **kwargs
                    )
                    auto_skim_memory(estimated_memory_mb, 'neural_net')
                    return original_multi_objective(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Memory skimming failed for multi-objective: {e}")
                    # Fallback to original method
                    return original_multi_objective(self, *args, **kwargs)

            # Replace method
            HyperparameterOptimization.multi_objective_optimization = enhanced_multi_objective_optimization

        if original_early_stopping is not None:
            def enhanced_early_stopping_optimization(self, *args, **kwargs):
                try:
                    manager = get_ml_memory_manager()
                    estimated_memory_mb = manager.estimate_ml_memory_requirements(
                        'hyperparameter_optimization', **kwargs
                    )
                    auto_skim_memory(estimated_memory_mb, 'neural_net')
                    return original_early_stopping(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Memory skimming failed for early stopping: {e}")
                    # Fallback to original method
                    return original_early_stopping(self, *args, **kwargs)

            # Replace method
            HyperparameterOptimization.early_stopping_optimization = enhanced_early_stopping_optimization

        logger.info("✅ Memory skimming integrated with HPO utilities")
        return True

    except Exception as e:
        logger.error(f"❌ HPO memory integration failed: {e}")
        return False

def integrate_memory_skimming_with_cv():
    """Integrate memory skimming with cross-validation utilities."""
    try:
        # Use unified CV instead of legacy cv_utils
        from ..validation.unified_cv import UnifiedCrossValidator
        logger.info("✅ Successfully imported Unified CV utilities")
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with CV utilities: {e}")
        return False

    try:
        # Add walk_forward-like helper onto UnifiedCrossValidator for compatibility
        if hasattr(UnifiedCrossValidator, 'run'):
            original_run = UnifiedCrossValidator.run

            def enhanced_run(self, model, X, y, *args, strategy: str = 'standard', **kwargs):
                try:
                    manager = get_ml_memory_manager()
                    estimated_memory_mb = manager.estimate_ml_memory_requirements(
                        'cross_validation', **kwargs
                    )
                    auto_skim_memory(estimated_memory_mb, 'data_processing')
                    return original_run(self, model, X, y, *args, strategy=strategy, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Memory skimming failed for CV run: {e}")
                    # Fallback to original method
                    return original_run(self, model, X, y, *args, strategy=strategy, **kwargs)

            # Replace method
            UnifiedCrossValidator.run = enhanced_run
            logger.info("✅ Memory skimming integrated with Unified CV utilities")
            return True
        else:
            logger.warning("⚠️ UnifiedCrossValidator.run method not found")
            return False

    except Exception as e:
        logger.error(f"❌ CV memory integration failed: {e}")
        return False

def integrate_memory_skimming_with_lookahead():
    """Integrate memory skimming with lookahead protection utilities."""
    # Initialize variables to avoid scoping issues
    original_temporal_validation = None

    try:
        from .lookahead_protection import LookaheadProtection

        # Store original method before enhancement
        original_temporal_validation = LookaheadProtection.temporal_feature_validation

        def enhanced_temporal_feature_validation(self, *args, **kwargs):
            # Check if original method was successfully captured
            if original_temporal_validation is None:
                logger.error("❌ Original temporal validation method not available")
                return None

            try:
                manager = get_ml_memory_manager()
                estimated_memory_mb = manager.estimate_ml_memory_requirements(
                    'lookahead_validation', **kwargs
                )
                auto_skim_memory(estimated_memory_mb, 'data_processing')
                return original_temporal_validation(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ Memory skimming failed for temporal validation: {e}")
                # Fallback to original method
                return original_temporal_validation(self, *args, **kwargs)

        # Replace method
        LookaheadProtection.temporal_feature_validation = enhanced_temporal_feature_validation

        logger.info("✅ Memory skimming integrated with lookahead protection utilities")
        return True

    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with lookahead protection utilities: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Lookahead integration failed: {e}")
        return False

def integrate_memory_skimming_with_model_evaluation():
    """Deprecated: ModelEvaluationUtilities removed; no integration needed."""
    logger.info("ℹ️ Skipping model evaluation memory integration (deprecated)")
    return True

def integrate_memory_skimming_with_feature_selection():
    """Integrate memory skimming with feature selection utilities."""
    try:
        from .feature_selection import FeatureSelectionFramework
        logger.info("✅ Successfully imported feature selection utilities")
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with feature selection utilities: {e}")
        return False

    try:
        # Add memory skimming to key methods
        if hasattr(FeatureSelectionFramework, 'select_features'):
            original_select_features = FeatureSelectionFramework.select_features

            def enhanced_select_features(self, *args, **kwargs):
                try:
                    manager = get_ml_memory_manager()
                    estimated_memory_mb = manager.estimate_ml_memory_requirements(
                        'feature_engineering', **kwargs
                    )
                    auto_skim_memory(estimated_memory_mb, 'feature_engineering')
                    return original_select_features(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Memory skimming failed for feature selection: {e}")
                    # Fallback to original method
                    return original_select_features(self, *args, **kwargs)

            # Replace method
            FeatureSelectionFramework.select_features = enhanced_select_features
            logger.info("✅ Memory skimming integrated with feature selection utilities")
            return True
        else:
            logger.warning("⚠️ FeatureSelectionFramework.select_features method not found")
            return False

    except Exception as e:
        logger.error(f"❌ Feature selection memory integration failed: {e}")
        return False

def integrate_memory_skimming_with_data_quality():
    """Integrate memory skimming with data quality utilities."""
    try:
        from .data_quality import DataQualityUtilities
        logger.info("✅ Successfully imported data quality utilities")
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with data quality utilities: {e}")
        return False

    try:
        # Add memory skimming to key methods
        if hasattr(DataQualityUtilities, 'automated_data_cleaning'):
            original_automated_cleaning = DataQualityUtilities.automated_data_cleaning

            def enhanced_automated_data_cleaning(self, *args, **kwargs):
                try:
                    manager = get_ml_memory_manager()
                    estimated_memory_mb = manager.estimate_ml_memory_requirements(
                        'data_preprocessing', **kwargs
                    )
                    auto_skim_memory(estimated_memory_mb, 'data_preprocessing')
                    return original_automated_cleaning(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Memory skimming failed for data cleaning: {e}")
                    # Fallback to original method
                    return original_automated_cleaning(self, *args, **kwargs)

            # Replace method
            DataQualityUtilities.automated_data_cleaning = enhanced_automated_data_cleaning
            logger.info("✅ Memory skimming integrated with data quality utilities")
            return True
        else:
            logger.warning("⚠️ DataQualityUtilities.automated_data_cleaning method not found")
            return False

    except Exception as e:
        logger.error(f"❌ Data quality memory integration failed: {e}")
        return False

def integrate_all_ml_utilities():
    """Integrate memory skimming with all ML utilities."""
    logger.info("🔗 Integrating memory skimming with all ML utilities")

    results = {
        'hpo_integration': integrate_memory_skimming_with_hpo(),
        'cv_integration': integrate_memory_skimming_with_cv(),
        'lookahead_integration': integrate_memory_skimming_with_lookahead(),
        'model_evaluation_integration': integrate_memory_skimming_with_model_evaluation(),
        'feature_selection_integration': integrate_memory_skimming_with_feature_selection(),
        'data_quality_integration': integrate_memory_skimming_with_data_quality()
    }

    successful_integrations = sum(results.values())
    total_integrations = len(results)

    logger.info(f"✅ Memory skimming integration completed: {successful_integrations}/{total_integrations} successful")

    return results

# Auto-integration on import
try:
    integrate_all_ml_utilities()
except Exception as e:
    logger.warning(f"⚠️ Auto-integration failed: {e}")

# Alias for backward compatibility
MemoryIntegrator = MLMemoryManager

# Export key functions and classes
__all__ = [
    'MLMemoryManager', 'MemoryIntegrator', 'get_ml_memory_manager',
    'ml_memory_skim_decorator', 'ml_auto_memory_skim_decorator',
    'ml_memory_context', 'ml_auto_memory_context',
    'integrate_memory_skimming_with_hpo', 'integrate_memory_skimming_with_cv',
    'integrate_memory_skimming_with_lookahead', 'integrate_memory_skimming_with_model_evaluation',
    'integrate_memory_skimming_with_feature_selection', 'integrate_memory_skimming_with_data_quality',
    'integrate_all_ml_utilities'
]
