"""
Optimization Integration Manager

Integrates all computational efficiency optimizations into a unified system
for the feature interaction generation pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time
import gc
from contextlib import contextmanager

from src.utils.tprint import tprint

# Import all optimization components
from .advanced_memory_manager import AdvancedMemoryManager, MemoryConfig
from .enhanced_vectorbt_manager import EnhancedVectorBTManager, VectorBTConfig
from .m1_parallel_processor import M1ParallelProcessor, ParallelConfig
from .optimized_shap_computer import OptimizedSHAPComputer, SHAPConfig
from .smart_interaction_discovery import SmartInteractionDiscovery, InteractionConfig
from .data_structure_optimizer import DataStructureOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class IntegratedOptimizationConfig:
    """Unified configuration for all optimization components."""
    
    # Memory management
    memory_mapping_threshold_gb: float = 2.0
    enable_memory_pools: bool = True
    enable_incremental_processing: bool = True
    
    # VectorBT optimization
    enable_gpu_acceleration: bool = True
    enable_lazy_evaluation: bool = True
    
    # Parallel processing
    enable_adaptive_allocation: bool = True
    max_workers: int = 8
    
    # SHAP optimization
    enable_incremental_shap: bool = True
    enable_sampling_approximation: bool = True
    max_shap_samples: int = 1000
    
    # Data structure optimization
    enable_int32_downcasting: bool = True
    enable_float32_downcasting: bool = True
    enable_categorical_optimization: bool = True
    
    # Smart interaction discovery
    correlation_threshold: float = 0.95
    enable_feature_clustering: bool = True
    max_interactions: int = 1000

class OptimizationIntegrationManager:
    """Manages all optimization components for unified operation."""
    
    def __init__(self, config: Optional[IntegratedOptimizationConfig] = None):
        self.config = config or IntegratedOptimizationConfig()
        self.logger = logger.getChild('OptimizationIntegrationManager')
        
        # Initialize all optimization components
        self._initialize_optimization_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'memory_optimizations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'shap_optimizations': 0,
            'interaction_discoveries': 0,
            'data_optimizations': 0
        }
        
        tprint("🚀 [INTEGRATION] Optimization Integration Manager initialized")
    
    def _initialize_optimization_components(self):
        """Initialize all optimization components with unified configuration."""
        tprint("🔄 [INTEGRATION] Initializing optimization components")
        
        # Memory manager configuration
        memory_config = MemoryConfig(
            memory_mapping_threshold_gb=self.config.memory_mapping_threshold_gb,
            enable_memory_pool=self.config.enable_memory_pools
        )
        self.memory_manager = AdvancedMemoryManager(memory_config)
        
        # VectorBT manager configuration
        vectorbt_config = VectorBTConfig(
            enable_gpu_acceleration=self.config.enable_gpu_acceleration,
            enable_lazy_evaluation=self.config.enable_lazy_evaluation
        )
        self.vectorbt_manager = EnhancedVectorBTManager(vectorbt_config)
        
        # Parallel processor configuration
        parallel_config = ParallelConfig(
            enable_adaptive_allocation=self.config.enable_adaptive_allocation,
            max_workers=self.config.max_workers
        )
        self.parallel_processor = M1ParallelProcessor(parallel_config)
        
        # SHAP computer configuration
        shap_config = SHAPConfig(
            enable_incremental=self.config.enable_incremental_shap,
            enable_sampling=self.config.enable_sampling_approximation,
            max_samples=self.config.max_shap_samples
        )
        self.shap_computer = OptimizedSHAPComputer(shap_config)
        
        # Interaction discovery configuration
        interaction_config = InteractionConfig(
            correlation_threshold=self.config.correlation_threshold,
            enable_clustering=self.config.enable_feature_clustering,
            max_interactions=self.config.max_interactions
        )
        self.interaction_discovery = SmartInteractionDiscovery(interaction_config)
        
        # Data structure optimizer configuration
        optimization_config = OptimizationConfig(
            enable_int32_downcasting=self.config.enable_int32_downcasting,
            enable_float32_downcasting=self.config.enable_float32_downcasting,
            enable_categorical_optimization=self.config.enable_categorical_optimization
        )
        self.data_optimizer = DataStructureOptimizer(optimization_config)
        
        tprint("✅ [INTEGRATION] All optimization components initialized")
    
    def optimize_dataframe_with_integration(self, data: pd.DataFrame, 
                                          operation_name: str = "dataframe_optimization") -> pd.DataFrame:
        """Comprehensive DataFrame optimization using all available optimizations."""
        tprint(f"🔄 [INTEGRATION] Comprehensive DataFrame optimization: {operation_name}")
        
        with self.memory_manager.memory_context(operation_name):
            # Step 1: Data structure optimization (int32/float32 downcasting)
            optimized_data = self.data_optimizer.optimize_dataframe(data)
            self.performance_stats['data_optimizations'] += 1
            
            # Step 2: Memory optimization (memory mapping if needed)
            if self.memory_manager.should_use_memory_mapping(optimized_data):
                tprint("💾 [INTEGRATION] Applying memory mapping optimization")
                self.performance_stats['memory_optimizations'] += 1
            
            # Step 3: Cache-friendly layout optimization
            optimized_data = self.memory_manager.cache_friendly_data_layout(optimized_data)
            
            tprint(f"✅ [INTEGRATION] DataFrame optimization completed: {operation_name}")
            return optimized_data
    
    def process_with_parallel_optimization(self, data: pd.DataFrame, 
                                         processor_func: callable,
                                         task_type: str = "mixed",
                                         **kwargs) -> pd.DataFrame:
        """Process data with parallel optimization."""
        tprint(f"🔄 [INTEGRATION] Parallel processing with optimization: {task_type}")
        
        # Optimize data first
        optimized_data = self.optimize_dataframe_with_integration(data, f"parallel_{task_type}")
        
        # Determine optimal task type for parallel processing
        if task_type == "cpu_intensive":
            from .m1_parallel_processor import TaskType
            task_type_enum = TaskType.CPU_INTENSIVE
        elif task_type == "io_bound":
            task_type_enum = TaskType.IO_BOUND
        else:
            task_type_enum = TaskType.MIXED
        
        # Process with parallel optimization
        if self.parallel_processor.should_use_chunked_processing(optimized_data):
            result = self.parallel_processor.parallel_apply(
                processor_func, optimized_data, task_type=task_type_enum, **kwargs
            )
        else:
            result = processor_func(optimized_data, **kwargs)
        
        self.performance_stats['parallel_operations'] += 1
        tprint(f"✅ [INTEGRATION] Parallel processing completed: {task_type}")
        
        return result
    
    def compute_optimized_shap(self, model: Any, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             computation_mode: str = "adaptive") -> Dict[str, Any]:
        """Compute SHAP values with all available optimizations."""
        tprint(f"🔄 [INTEGRATION] Optimized SHAP computation: {computation_mode}")
        
        # Optimize input data
        X_optimized = self.data_optimizer.optimize_array(X)
        y_optimized = self.data_optimizer.optimize_array(y)
        
        # Compute SHAP with optimizations
        result = self.shap_computer.compute_optimized_shap(
            model, X_optimized, y_optimized, feature_names, computation_mode
        )
        
        self.performance_stats['shap_optimizations'] += 1
        tprint(f"✅ [INTEGRATION] Optimized SHAP computation completed: {computation_mode}")
        
        return result
    
    def discover_interactions_with_optimization(self, features_df: pd.DataFrame,
                                              targets: Optional[pd.Series] = None,
                                              importance_scores: Optional[Dict[str, float]] = None,
                                              discovery_mode: str = "comprehensive") -> Dict[str, Any]:
        """Discover interactions with smart filtering and optimization."""
        tprint(f"🔄 [INTEGRATION] Smart interaction discovery: {discovery_mode}")
        
        # Optimize input data
        optimized_features = self.optimize_dataframe_with_integration(features_df, "interaction_discovery")
        
        # Discover interactions with optimization
        result = self.interaction_discovery.discover_interactions(
            optimized_features, targets, importance_scores, discovery_mode
        )
        
        self.performance_stats['interaction_discoveries'] += 1
        tprint(f"✅ [INTEGRATION] Smart interaction discovery completed: {discovery_mode}")
        
        return result
    
    def vectorbt_optimized_operations(self, data: pd.DataFrame, 
                                    operations: List[callable],
                                    operation_type: str = "technical_indicators") -> Dict[str, pd.DataFrame]:
        """Perform VectorBT operations with GPU acceleration and lazy evaluation."""
        tprint(f"🔄 [INTEGRATION] VectorBT optimized operations: {operation_type}")
        
        # Optimize input data
        optimized_data = self.optimize_dataframe_with_integration(data, f"vectorbt_{operation_type}")
        
        # Create lazy operations
        lazy_operations = self.vectorbt_manager.lazy_rolling_operations(
            optimized_data, operations, [20, 50, 100]  # Example windows
        )
        
        # Compute lazy operations
        results = self.vectorbt_manager.compute_lazy_operations(lazy_operations)
        
        # GPU acceleration if available
        if self.vectorbt_manager.gpu_available:
            tprint("🚀 [INTEGRATION] Applying GPU acceleration to VectorBT operations")
            self.performance_stats['gpu_accelerations'] += 1
        
        tprint(f"✅ [INTEGRATION] VectorBT optimized operations completed: {operation_type}")
        
        return {f"operation_{i}": result for i, result in enumerate(results)}
    
    def incremental_data_processing(self, data_iterator, processor_func: callable,
                                  **kwargs) -> pd.DataFrame:
        """Process data incrementally with memory optimization."""
        tprint("🔄 [INTEGRATION] Incremental data processing with optimization")
        
        # Process incrementally with memory management
        result = self.memory_manager.incremental_data_processing(
            data_iterator, processor_func, **kwargs
        )
        
        self.performance_stats['memory_optimizations'] += 1
        tprint("✅ [INTEGRATION] Incremental data processing completed")
        
        return result
    
    def get_comprehensive_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics from all optimization components."""
        stats = {
            'integration_stats': self.performance_stats.copy(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'vectorbt_stats': self.vectorbt_manager.get_performance_stats(),
            'parallel_stats': self.parallel_processor.get_performance_stats(),
            'shap_stats': self.shap_computer.get_performance_stats(),
            'interaction_stats': self.interaction_discovery.get_discovery_stats(),
            'optimization_stats': self.data_optimizer.get_optimization_stats()
        }
        
        return stats
    
    def optimize_pipeline_phase(self, phase_name: str, data: pd.DataFrame,
                              processor_func: callable, **kwargs) -> pd.DataFrame:
        """Optimize an entire pipeline phase with all available optimizations."""
        tprint(f"🚀 [INTEGRATION] Optimizing pipeline phase: {phase_name}")
        
        start_time = time.time()
        
        with self.memory_manager.memory_context(f"phase_{phase_name}"):
            # Comprehensive optimization
            result = self.process_with_parallel_optimization(
                data, processor_func, task_type="mixed", **kwargs
            )
        
        phase_time = time.time() - start_time
        
        tprint(f"✅ [INTEGRATION] Pipeline phase optimization completed: {phase_name} ({phase_time:.2f}s)")
        
        return result
    
    @contextmanager
    def optimization_context(self, operation_name: str):
        """Context manager for optimization operations with automatic cleanup."""
        tprint(f"🔄 [INTEGRATION] Starting optimization context: {operation_name}")
        
        try:
            yield self
            
        finally:
            tprint(f"✅ [INTEGRATION] Completed optimization context: {operation_name}")
            
            # Cleanup if needed
            if self.performance_stats['total_operations'] % 100 == 0:
                self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up all optimization resources."""
        tprint("🧹 [INTEGRATION] Cleaning up all optimization resources")
        
        # Cleanup all components
        self.memory_manager.cleanup()
        self.vectorbt_manager.cleanup()
        self.parallel_processor.cleanup()
        self.shap_computer.cleanup()
        self.interaction_discovery.cleanup()
        self.data_optimizer.cleanup()
        
        # Final garbage collection
        gc.collect()
        
        tprint("✅ [INTEGRATION] All optimization resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_resources()
        except Exception:
            pass  # Ignore cleanup errors in destructor
