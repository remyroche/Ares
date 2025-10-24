"""
Parallel Training Manager with Hardware Optimization

This module provides parallel training capabilities using hardware optimization utilities
for efficient resource management and performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

# Hardware optimization imports
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    OptimizationConfig, OptimizationLevel
)
from src.utils.hardware.integrated_hardware_manager import (
    IntegratedHardwareManager, IntegratedHardwareConfig
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel
)
from src.utils.hardware.dynamic_memory_allocator import (
    get_dynamic_allocator, get_optimal_memory_allocation, WorkloadType
)

from .error_handling import (
    handle_errors, ErrorContext,
    MLModelTrainerError, ResourceError, ModelTrainingError
)

logger = logging.getLogger(__name__)

class TrainingStrategy(Enum):
    """Training strategies for parallel execution."""
    SEQUENTIAL = "sequential"
    THREAD_PARALLEL = "thread_parallel"
    PROCESS_PARALLEL = "process_parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training."""
    strategy: TrainingStrategy = TrainingStrategy.ADAPTIVE
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    gpu_memory_limit_gb: float = 4.0
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    chunk_size: int = 1000
    timeout_seconds: int = 3600
    retry_attempts: int = 3
    resource_monitoring: bool = True

class ParallelTrainingManager:
    """Manages parallel training with hardware optimization."""
    
    def __init__(self, config: ParallelTrainingConfig = None):
        self.config = config or ParallelTrainingConfig()
        
        # Initialize hardware manager
        hardware_config = IntegratedHardwareConfig(
            enable_caching=self.config.enable_caching,
            enable_memory_optimization=self.config.enable_memory_optimization,
            memory_limit_gb=self.config.memory_limit_gb,
            gpu_memory_limit_gb=self.config.gpu_memory_limit_gb
        )
        self.hardware_manager = IntegratedHardwareManager(hardware_config)
        
        # Initialize dynamic memory allocator
        self.memory_allocator = get_dynamic_allocator()
        
        # Training state
        self.active_tasks = {}
        self.resource_usage = {}
        self.performance_metrics = {}
    
    @performance_tracked(level=OptimizationLevel.HIGH)
    @memory_optimized(level=MemoryOptimizationLevel.AGGRESSIVE)
    async def train_models_parallel(
        self,
        models: List[Dict[str, Any]],
        data: Dict[str, Any],
        configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train multiple models in parallel with hardware optimization.
        
        Args:
            models: List of model configurations
            data: Training data dictionary
            configs: Model-specific configurations
            
        Returns:
            Dictionary of training results
        """
        with ErrorContext("parallel_training", self._cleanup_training_resources):
            tprint_info(f"Starting parallel training of {len(models)} models")
            logger.info(f"Starting parallel training of {len(models)} models")
            
            # Determine optimal strategy
            strategy = self._determine_optimal_strategy(models, data)
            tprint_info(f"Using training strategy: {strategy}")
            logger.info(f"Using training strategy: {strategy}")
            
            # Allocate resources
            tprint_debug("Allocating training resources")
            await self._allocate_resources(models, data)
            
            # Execute training based on strategy
            if strategy == TrainingStrategy.SEQUENTIAL:
                tprint_info("Executing sequential training")
                results = await self._train_sequential(models, data, configs)
            elif strategy == TrainingStrategy.THREAD_PARALLEL:
                tprint_info("Executing thread-parallel training")
                results = await self._train_thread_parallel(models, data, configs)
            elif strategy == TrainingStrategy.PROCESS_PARALLEL:
                tprint_info("Executing process-parallel training")
                results = await self._train_process_parallel(models, data, configs)
            elif strategy == TrainingStrategy.HYBRID:
                tprint_info("Executing hybrid training")
                results = await self._train_hybrid(models, data, configs)
            else:  # ADAPTIVE
                tprint_info("Executing adaptive training")
                results = await self._train_adaptive(models, data, configs)
            
            # Cleanup and return results
            tprint_debug("Cleaning up training resources")
            await self._cleanup_training_resources()
            tprint_success("Parallel training completed")
            return results
    
    def _determine_optimal_strategy(
        self, 
        models: List[Dict[str, Any]], 
        data: Dict[str, Any]
    ) -> TrainingStrategy:
        """Determine the optimal training strategy based on resources and data."""
        
        # Get system resources
        system_info = self.hardware_manager.get_system_info()
        available_memory = system_info.get('available_memory_gb', 8.0)
        cpu_cores = system_info.get('cpu_cores', 4)
        gpu_available = system_info.get('gpu_available', False)
        
        # Analyze data size
        data_size_gb = self._estimate_data_size(data)
        model_count = len(models)
        
        # Determine strategy based on resources and workload
        if model_count == 1:
            return TrainingStrategy.SEQUENTIAL
        elif data_size_gb > available_memory * 0.8:
            return TrainingStrategy.SEQUENTIAL  # Not enough memory for parallel
        elif model_count <= 2 and cpu_cores >= 4:
            return TrainingStrategy.THREAD_PARALLEL
        elif model_count > 2 and cpu_cores >= 8 and data_size_gb < available_memory * 0.5:
            return TrainingStrategy.PROCESS_PARALLEL
        elif gpu_available and any('gpu' in str(model).lower() for model in models):
            return TrainingStrategy.HYBRID
        else:
            return TrainingStrategy.ADAPTIVE
    
    def _estimate_data_size(self, data: Dict[str, Any]) -> float:
        """Estimate data size in GB."""
        total_size = 0
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            elif isinstance(value, pd.DataFrame):
                total_size += value.memory_usage(deep=True).sum()
            elif isinstance(value, dict):
                total_size += self._estimate_data_size(value)
        return total_size / (1024**3)
    
    async def _allocate_resources(self, models: List[Dict[str, Any]], data: Dict[str, Any]):
        """Allocate resources for parallel training."""
        
        # Get optimal memory allocation
        workload_type = WorkloadType.MACHINE_LEARNING_TRAINING
        allocation = get_optimal_memory_allocation(
            workload_type=workload_type,
            data_size_gb=self._estimate_data_size(data),
            model_count=len(models),
            available_memory_gb=self.config.memory_limit_gb
        )
        
        logger.info(f"Allocated memory: {allocation}")
        
        # Update memory limits based on allocation
        self.config.memory_limit_gb = allocation.get('memory_limit_gb', self.config.memory_limit_gb)
        self.config.max_workers = allocation.get('max_workers', self.config.max_workers)
    
    @smart_cache(ttl=3600, max_size=1000)
    @auto_optimize(level=OptimizationLevel.HIGH)
    async def _train_sequential(
        self, 
        models: List[Dict[str, Any]], 
        data: Dict[str, Any], 
        configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train models sequentially."""
        results = {}
        
        for i, model_config in enumerate(models):
            model_name = model_config.get('name', f'model_{i}')
            logger.info(f"Training {model_name} sequentially")
            
            try:
                with ErrorContext(f"training_{model_name}"):
                    result = await self._train_single_model(
                        model_config, data, configs.get(model_name, {})
                    )
                    results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    @performance_tracked(level=OptimizationLevel.HIGH)
    async def _train_thread_parallel(
        self, 
        models: List[Dict[str, Any]], 
        data: Dict[str, Any], 
        configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train models using thread parallelism."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all training tasks
            future_to_model = {}
            for i, model_config in enumerate(models):
                model_name = model_config.get('name', f'model_{i}')
                future = executor.submit(
                    self._train_single_model_sync,
                    model_config, data, configs.get(model_name, {})
                )
                future_to_model[future] = model_name
            
            # Collect results as they complete
            for future in as_completed(future_to_model, timeout=self.config.timeout_seconds):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    logger.info(f"Completed training {model_name}")
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
        
        return results
    
    @performance_tracked(level=OptimizationLevel.HIGH)
    async def _train_process_parallel(
        self, 
        models: List[Dict[str, Any]], 
        data: Dict[str, Any], 
        configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train models using process parallelism."""
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all training tasks
            future_to_model = {}
            for i, model_config in enumerate(models):
                model_name = model_config.get('name', f'model_{i}')
                future = executor.submit(
                    self._train_single_model_sync,
                    model_config, data, configs.get(model_name, {})
                )
                future_to_model[future] = model_name
            
            # Collect results as they complete
            for future in as_completed(future_to_model, timeout=self.config.timeout_seconds):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    logger.info(f"Completed training {model_name}")
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
        
        return results
    
    async def _train_hybrid(
        self, 
        models: List[Dict[str, Any]], 
        data: Dict[str, Any], 
        configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train models using hybrid CPU/GPU parallelism."""
        # Separate CPU and GPU models
        cpu_models = []
        gpu_models = []
        
        for model_config in models:
            if 'gpu' in str(model_config).lower() or 'cuda' in str(model_config).lower():
                gpu_models.append(model_config)
            else:
                cpu_models.append(model_config)
        
        results = {}
        
        # Train CPU models in parallel
        if cpu_models:
            cpu_results = await self._train_thread_parallel(cpu_models, data, configs)
            results.update(cpu_results)
        
        # Train GPU models sequentially (GPU memory constraints)
        for model_config in gpu_models:
            model_name = model_config.get('name', 'gpu_model')
            try:
                result = await self._train_single_model(model_config, data, configs.get(model_name, {}))
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to train GPU model {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    async def _train_adaptive(
        self, 
        models: List[Dict[str, Any]], 
        data: Dict[str, Any], 
        configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train models using adaptive strategy based on real-time resource usage."""
        results = {}
        
        # Start with thread parallelism
        strategy = TrainingStrategy.THREAD_PARALLEL
        
        for i, model_config in enumerate(models):
            model_name = model_config.get('name', f'model_{i}')
            
            # Monitor resources before each model
            resource_usage = self.hardware_manager.get_resource_usage()
            
            # Adapt strategy based on resource usage
            if resource_usage.get('memory_usage_percent', 0) > 80:
                strategy = TrainingStrategy.SEQUENTIAL
            elif resource_usage.get('cpu_usage_percent', 0) < 50 and i > 0:
                strategy = TrainingStrategy.THREAD_PARALLEL
            
            # Train model with current strategy
            try:
                if strategy == TrainingStrategy.SEQUENTIAL:
                    result = await self._train_single_model(model_config, data, configs.get(model_name, {}))
                else:
                    # Use thread pool for this model
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            self._train_single_model_sync,
                            model_config, data, configs.get(model_name, {})
                        )
                        result = future.result(timeout=self.config.timeout_seconds)
                
                results[model_name] = result
                logger.info(f"Completed training {model_name} with {strategy.value}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    @handle_errors(error_type=ModelTrainingError, reraise=True)
    async def _train_single_model(
        self, 
        model_config: Dict[str, Any], 
        data: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single model with hardware optimization."""
        
        # Apply memory optimization
        with comprehensive_memory_optimization(level=MemoryOptimizationLevel.AGGRESSIVE):
            # This would integrate with the actual model training logic
            # For now, return a placeholder
            return {
                'model_type': model_config.get('type', 'unknown'),
                'status': 'trained',
                'timestamp': time.time()
            }
    
    def _train_single_model_sync(
        self, 
        model_config: Dict[str, Any], 
        data: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous version of single model training for parallel execution."""
        # This would integrate with the actual model training logic
        # For now, return a placeholder
        return {
            'model_type': model_config.get('type', 'unknown'),
            'status': 'trained',
            'timestamp': time.time()
        }
    
    async def _cleanup_training_resources(self):
        """Cleanup training resources."""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear caches if memory usage is high
        resource_usage = self.hardware_manager.get_resource_usage()
        if resource_usage.get('memory_usage_percent', 0) > 90:
            self.hardware_manager.clear_caches()
        
        logger.info("Training resources cleaned up")

# Factory function
def create_parallel_training_manager(config: ParallelTrainingConfig = None) -> ParallelTrainingManager:
    """Create a parallel training manager with configuration."""
    return ParallelTrainingManager(config)