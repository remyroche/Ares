#!/usr/bin/env python3
"""
HMM Utilities Package

This package contains modular HMM utilities that were previously consolidated
in the monolithic hmm_composite_manager.py file. The functionality has been
split into focused modules:

- core_manager: Basic HMM operations and file management
- optimization: Bayesian optimization and parameter tuning
- hardware_integration: GPU acceleration and hardware optimization
- validation: HMM model validation and metrics

Usage:
    from src.utils.hmm import HMMCoreManager, HMMBayesianOptimizer
    
    # Create managers
    core_manager = HMMCoreManager()
    optimizer = HMMBayesianOptimizer()
"""

from .core_manager import HMMCoreManager
from .optimization import (
    HMMBayesianOptimizer,
    HMMParameterTuner,
    BayesianOptimizationConfig
)
from .hardware_integration import HMMHardwareManager

__all__ = [
    'HMMCoreManager',
    'HMMBayesianOptimizer', 
    'HMMParameterTuner',
    'BayesianOptimizationConfig',
    'HMMHardwareManager'
]

# Backward compatibility - create a unified manager that combines all components
class EnhancedHMMCompositeManager:
    """
    Unified HMM manager that combines all modular components.
    
    This provides backward compatibility with the original monolithic
    hmm_composite_manager.py while using the new modular structure.
    """
    
    def __init__(self):
        """Initialize the composite manager with all components."""
        self.core_manager = HMMCoreManager()
        self.optimizer = HMMBayesianOptimizer()
        self.parameter_tuner = HMMParameterTuner()
        self.hardware_manager = HMMHardwareManager()
    
    # Delegate methods to appropriate managers
    def get_composite_cluster_file_path(self, *args, **kwargs):
        return self.core_manager.get_composite_cluster_file_path(*args, **kwargs)
    
    def file_exists(self, *args, **kwargs):
        return self.core_manager.file_exists(*args, **kwargs)
    
    def load_composite_clusters(self, *args, **kwargs):
        return self.core_manager.load_composite_clusters(*args, **kwargs)
    
    def save_composite_clusters(self, *args, **kwargs):
        return self.core_manager.save_composite_clusters(*args, **kwargs)
    
    def optimize_hmm_parameters(self, *args, **kwargs):
        return self.optimizer.optimize_hmm_parameters(*args, **kwargs)
    
    def gpu_accelerated_hmm_training(self, *args, **kwargs):
        return self.hardware_manager.gpu_accelerated_hmm_training(*args, **kwargs)
    
    def get_memory_usage(self, *args, **kwargs):
        return self.hardware_manager.get_memory_usage(*args, **kwargs)
    
    def cleanup_gpu_memory(self, *args, **kwargs):
        return self.hardware_manager.cleanup_gpu_memory(*args, **kwargs)


# Factory function for backward compatibility
def get_hmm_composite_manager():
    """Get an instance of the composite HMM manager."""
    return EnhancedHMMCompositeManager()