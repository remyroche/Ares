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
from src.utils.hardware.m1_memory_optimizer import (
    auto_skim_memory, smart_memory_allocation,
    memory_skim_decorator, auto_memory_skim_decorator,
    auto_memory_skim_context, smart_memory_context,
    get_m1_memory_optimizer
)

logger = logging.getLogger(__name__)

class MLMemoryManager:
    """Memory manager specifically designed for ML operations."""
    
    def __init__(self):
        self.logger = logger.getChild('MLMemoryManager')
        self.logger.info("🚀 Initializing MLMemoryManager...")
        start_time = time.time()
        
        self.memory_optimizer = get_m1_memory_optimizer()
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
    # Initialize variables to avoid scoping issues
    original_multi_objective = None
    original_early_stopping = None

    try:
        from .hpo_utils import HyperparameterOptimization

        # Store original methods before enhancement
        original_multi_objective = HyperparameterOptimization.multi_objective_optimization
        original_early_stopping = HyperparameterOptimization.early_stopping_optimization

        def enhanced_multi_objective_optimization(self, *args, **kwargs):
            # Check if original method was successfully captured
            if original_multi_objective is None:
                logger.error("❌ Original multi-objective method not available")
                return None

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

        def enhanced_early_stopping_optimization(self, *args, **kwargs):
            # Check if original method was successfully captured
            if original_early_stopping is None:
                logger.error("❌ Original early stopping method not available")
                return None

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

        # Replace methods
        HyperparameterOptimization.multi_objective_optimization = enhanced_multi_objective_optimization
        HyperparameterOptimization.early_stopping_optimization = enhanced_early_stopping_optimization

        logger.info("✅ Memory skimming integrated with HPO utilities")
        return True

    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with HPO utilities: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Memory integration failed: {e}")
        return False

def integrate_memory_skimming_with_cv():
    """Integrate memory skimming with cross-validation utilities."""
    # Initialize variables to avoid scoping issues
    original_walk_forward = None

    try:
        from .cv_utils import CrossValidationUtilities

        # Store original method before enhancement
        original_walk_forward = CrossValidationUtilities.walk_forward_validation

        def enhanced_walk_forward_validation(self, *args, **kwargs):
            # Check if original method was successfully captured
            if original_walk_forward is None:
                logger.error("❌ Original walk-forward method not available")
                return None

            try:
                manager = get_ml_memory_manager()
                estimated_memory_mb = manager.estimate_ml_memory_requirements(
                    'cross_validation', **kwargs
                )
                auto_skim_memory(estimated_memory_mb, 'data_processing')
                return original_walk_forward(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ Memory skimming failed for walk-forward: {e}")
                # Fallback to original method
                return original_walk_forward(self, *args, **kwargs)

        # Replace method
        CrossValidationUtilities.walk_forward_validation = enhanced_walk_forward_validation

        logger.info("✅ Memory skimming integrated with CV utilities")
        return True

    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with CV utilities: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ CV integration failed: {e}")
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
    """Integrate memory skimming with model evaluation utilities."""
    try:
        from .model_evaluation import ModelEvaluationUtilities
        
        # Add memory skimming to key methods
        original_multi_metric = ModelEvaluationUtilities.multi_metric_evaluation
        
        def enhanced_multi_metric_evaluation(self, *args, **kwargs):
            manager = get_ml_memory_manager()
            estimated_memory_mb = manager.estimate_ml_memory_requirements(
                'model_inference', **kwargs
            )
            auto_skim_memory(estimated_memory_mb, 'model_inference')
            return original_multi_metric(self, *args, **kwargs)
        
        # Replace method
        ModelEvaluationUtilities.multi_metric_evaluation = enhanced_multi_metric_evaluation
        
        logger.info("✅ Memory skimming integrated with model evaluation utilities")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with model evaluation utilities: {e}")
        return False

def integrate_memory_skimming_with_feature_selection():
    """Integrate memory skimming with feature selection utilities."""
    try:
        from .feature_selection import FeatureSelectionFramework
        
        # Add memory skimming to key methods
        original_select_features = FeatureSelectionFramework.select_features
        
        def enhanced_select_features(self, *args, **kwargs):
            manager = get_ml_memory_manager()
            estimated_memory_mb = manager.estimate_ml_memory_requirements(
                'feature_engineering', **kwargs
            )
            auto_skim_memory(estimated_memory_mb, 'feature_engineering')
            return original_select_features(self, *args, **kwargs)
        
        # Replace method
        FeatureSelectionFramework.select_features = enhanced_select_features
        
        logger.info("✅ Memory skimming integrated with feature selection utilities")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with feature selection utilities: {e}")
        return False

def integrate_memory_skimming_with_data_quality():
    """Integrate memory skimming with data quality utilities."""
    try:
        from .data_quality import DataQualityUtilities
        
        # Add memory skimming to key methods
        original_automated_cleaning = DataQualityUtilities.automated_data_cleaning
        
        def enhanced_automated_data_cleaning(self, *args, **kwargs):
            manager = get_ml_memory_manager()
            estimated_memory_mb = manager.estimate_ml_memory_requirements(
                'data_preprocessing', **kwargs
            )
            auto_skim_memory(estimated_memory_mb, 'data_preprocessing')
            return original_automated_cleaning(self, *args, **kwargs)
        
        # Replace method
        DataQualityUtilities.automated_data_cleaning = enhanced_automated_data_cleaning
        
        logger.info("✅ Memory skimming integrated with data quality utilities")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Could not integrate with data quality utilities: {e}")
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

# Export key functions and classes
__all__ = [
    'MLMemoryManager', 'get_ml_memory_manager',
    'ml_memory_skim_decorator', 'ml_auto_memory_skim_decorator',
    'ml_memory_context', 'ml_auto_memory_context',
    'integrate_memory_skimming_with_hpo', 'integrate_memory_skimming_with_cv',
    'integrate_memory_skimming_with_lookahead', 'integrate_memory_skimming_with_model_evaluation',
    'integrate_memory_skimming_with_feature_selection', 'integrate_memory_skimming_with_data_quality',
    'integrate_all_ml_utilities'
]
