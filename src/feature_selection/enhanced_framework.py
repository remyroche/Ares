"""
Enhanced Feature Selection Framework

This module provides an enhanced feature selection framework that integrates
all the performance optimizations, error handling, caching, and hardware
optimization capabilities.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import all enhanced capabilities
from .caching import FeatureSelectionCacheManager, CacheConfig
from .error_handling import EnhancedErrorHandler, robust_feature_selection
from .memory import MemoryEfficientFeatureSelector, MemoryConfig
from .parallel import ParallelFeatureSelector, ParallelConfig
from .optimizations import VectorizedFeatureSelector, VectorizationConfig
from .sparse import SparseFeatureSelector, SparseConfig
from .chunked import ChunkedFeatureProcessor, ChunkedConfig

# Import core framework
from .core.framework import select_features as core_select_features

from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_info

logger = logging.getLogger(__name__)

@dataclass
class EnhancedFeatureSelectionConfig:
    """Configuration for enhanced feature selection framework."""
    # Enable/disable features
    enable_caching: bool = True
    enable_error_handling: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_vectorization: bool = True
    enable_sparse_support: bool = True
    enable_chunked_processing: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_limit_gb: float = 8.0
    
    # Performance settings
    auto_select_optimizations: bool = True
    enable_performance_monitoring: bool = True
    
    # Cache configuration
    cache_config: Optional[CacheConfig] = None
    
    # Memory configuration
    memory_config: Optional[MemoryConfig] = None
    
    # Parallel configuration
    parallel_config: Optional[ParallelConfig] = None
    
    # Vectorization configuration
    vectorization_config: Optional[VectorizationConfig] = None
    
    # Sparse configuration
    sparse_config: Optional[SparseConfig] = None
    
    # Chunked configuration
    chunked_config: Optional[ChunkedConfig] = None

class EnhancedFeatureSelectionFramework:
    """Enhanced feature selection framework with all optimizations."""
    
    def __init__(self, config: Optional[EnhancedFeatureSelectionConfig] = None):
        """Initialize enhanced framework."""
        self.config = config or EnhancedFeatureSelectionConfig()
        self.logger = logger.getChild('EnhancedFeatureSelectionFramework')
        
        # Initialize components
        self._initialize_components()
        
        tprint_success("🚀 EnhancedFeatureSelectionFramework initialized")
    
    def _initialize_components(self):
        """Initialize all framework components."""
        # Initialize cache manager
        if self.config.enable_caching:
            cache_config = self.config.cache_config or CacheConfig(
                enable_hardware_optimization=self.config.enable_hardware_optimization,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.cache_manager = FeatureSelectionCacheManager(cache_config)
        else:
            self.cache_manager = None
        
        # Initialize error handler
        if self.config.enable_error_handling:
            self.error_handler = EnhancedErrorHandler()
        else:
            self.error_handler = None
        
        # Initialize memory-efficient selector
        if self.config.enable_memory_optimization:
            memory_config = self.config.memory_config or MemoryConfig(
                memory_limit_gb=self.config.memory_limit_gb,
                enable_hardware_optimization=self.config.enable_hardware_optimization
            )
            self.memory_selector = MemoryEfficientFeatureSelector(memory_config)
        else:
            self.memory_selector = None
        
        # Initialize parallel selector
        if self.config.enable_parallel_processing:
            parallel_config = self.config.parallel_config or ParallelConfig(
                enable_hardware_optimization=self.config.enable_hardware_optimization
            )
            self.parallel_selector = ParallelFeatureSelector(parallel_config)
        else:
            self.parallel_selector = None
        
        # Initialize vectorized selector
        if self.config.enable_vectorization:
            vectorization_config = self.config.vectorization_config or VectorizationConfig(
                enable_hardware_acceleration=self.config.enable_hardware_optimization
            )
            self.vectorized_selector = VectorizedFeatureSelector(vectorization_config)
        else:
            self.vectorized_selector = None
        
        # Initialize sparse selector
        if self.config.enable_sparse_support:
            sparse_config = self.config.sparse_config or SparseConfig(
                memory_limit_gb=self.config.memory_limit_gb,
                enable_memory_monitoring=self.config.enable_hardware_optimization
            )
            self.sparse_selector = SparseFeatureSelector(sparse_config)
        else:
            self.sparse_selector = None
        
        # Initialize chunked processor
        if self.config.enable_chunked_processing:
            chunked_config = self.config.chunked_config or ChunkedConfig(
                memory_limit_gb=self.config.memory_limit_gb,
                enable_hardware_optimization=self.config.enable_hardware_optimization
            )
            self.chunked_processor = ChunkedFeatureProcessor(chunked_config)
        else:
            self.chunked_processor = None
    
    def _auto_select_optimizations(self, X: Union[np.ndarray, pd.DataFrame], 
                                 y: Union[np.ndarray, pd.Series]) -> Dict[str, bool]:
        """Automatically select optimizations based on data characteristics."""
        if not self.config.auto_select_optimizations:
            return {
                'use_caching': self.config.enable_caching,
                'use_memory_optimization': self.config.enable_memory_optimization,
                'use_parallel_processing': self.config.enable_parallel_processing,
                'use_vectorization': self.config.enable_vectorization,
                'use_sparse_support': self.config.enable_sparse_support,
                'use_chunked_processing': self.config.enable_chunked_processing
            }
        
        # Analyze data characteristics
        n_samples, n_features = X.shape if hasattr(X, 'shape') else (len(X), 0)
        
        # Determine optimizations
        optimizations = {
            'use_caching': self.config.enable_caching and n_samples > 1000,
            'use_memory_optimization': self.config.enable_memory_optimization and n_samples > 10000,
            'use_parallel_processing': self.config.enable_parallel_processing and n_features > 50,
            'use_vectorization': self.config.enable_vectorization and n_features > 20,
            'use_sparse_support': self.config.enable_sparse_support and self._is_sparse_beneficial(X),
            'use_chunked_processing': self.config.enable_chunked_processing and n_samples > 50000
        }
        
        tprint_info(f"🔧 Auto-selected optimizations: {[k for k, v in optimizations.items() if v]}")
        
        return optimizations
    
    def _is_sparse_beneficial(self, X: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check if sparse representation would be beneficial."""
        if hasattr(X, 'sparse') or hasattr(X, 'nnz'):
            return True
        
        # Check sparsity
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        zero_ratio = np.count_nonzero(X_array == 0) / X_array.size
        return zero_ratio > 0.1  # Use sparse if >10% zeros
    
    def select_features(self, X: Union[np.ndarray, pd.DataFrame], 
                       y: Union[np.ndarray, pd.Series],
                       method: str = 'comprehensive',
                       **kwargs) -> Dict[str, Any]:
        """Enhanced feature selection with automatic optimization selection."""
        tprint_info(f"🔍 Starting enhanced feature selection: {method}")
        
        start_time = time.time()
        
        try:
            # Auto-select optimizations
            optimizations = self._auto_select_optimizations(X, y)
            
            # Check cache first
            if optimizations['use_caching'] and self.cache_manager:
                cached_result = self.cache_manager.get_cached_selection(X, y, method, kwargs)
                if cached_result:
                    tprint_success("💾 Using cached result")
                    return cached_result
            
            # Select appropriate processing method
            if optimizations['use_chunked_processing'] and self.chunked_processor:
                # Use chunked processing for very large datasets
                result = self._process_with_chunked(X, y, method, **kwargs)
            elif optimizations['use_sparse_support'] and self.sparse_selector:
                # Use sparse processing for sparse data
                result = self._process_with_sparse(X, y, method, **kwargs)
            elif optimizations['use_memory_optimization'] and self.memory_selector:
                # Use memory-efficient processing
                result = self._process_with_memory_optimization(X, y, method, **kwargs)
            elif optimizations['use_vectorization'] and self.vectorized_selector:
                # Use vectorized processing
                result = self._process_with_vectorization(X, y, method, **kwargs)
            else:
                # Use core framework
                result = self._process_with_core(X, y, method, **kwargs)
            
            # Cache result if enabled
            if optimizations['use_caching'] and self.cache_manager and result.get('success', False):
                self.cache_manager.cache_selection_result(X, y, method, kwargs, result)
            
            # Add performance metrics
            end_time = time.time()
            result['execution_time'] = end_time - start_time
            result['optimizations_used'] = [k for k, v in optimizations.items() if v]
            
            tprint_success(f"✅ Enhanced selection completed: {len(result.get('selected_features', []))} features "
                         f"in {result['execution_time']:.3f}s")
            
            return result
            
        except Exception as e:
            # Use error handler if available
            if self.error_handler:
                context = {
                    'operation': 'enhanced_feature_selection',
                    'method': method,
                    'data_shape': X.shape if hasattr(X, 'shape') else (len(X), 0),
                    'parameters': kwargs
                }
                return self.error_handler.handle_error(e, context)
            else:
                raise e
    
    def _process_with_chunked(self, X: Union[np.ndarray, pd.DataFrame], 
                            y: Union[np.ndarray, pd.Series],
                            method: str, **kwargs) -> Dict[str, Any]:
        """Process using chunked processing."""
        tprint_info("📦 Using chunked processing")
        
        def processor_func(X_chunk, y_chunk, **kwargs):
            return self._process_with_core(X_chunk, y_chunk, method, **kwargs)
        
        return self.chunked_processor.process_large_dataset(X, y, processor_func, **kwargs)
    
    def _process_with_sparse(self, X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series],
                           method: str, **kwargs) -> Dict[str, Any]:
        """Process using sparse matrix operations."""
        tprint_info("📊 Using sparse processing")
        
        return self.sparse_selector.select_features_sparse(X, y, method, **kwargs)
    
    def _process_with_memory_optimization(self, X: Union[np.ndarray, pd.DataFrame], 
                                        y: Union[np.ndarray, pd.Series],
                                        method: str, **kwargs) -> Dict[str, Any]:
        """Process using memory optimization."""
        tprint_info("🧠 Using memory optimization")
        
        return self.memory_selector.select_features_chunked(X, y, method, **kwargs)
    
    def _process_with_vectorization(self, X: Union[np.ndarray, pd.DataFrame], 
                                  y: Union[np.ndarray, pd.Series],
                                  method: str, **kwargs) -> Dict[str, Any]:
        """Process using vectorized operations."""
        tprint_info("🚀 Using vectorized processing")
        
        return self.vectorized_selector.vectorized_feature_selection(X, y, method, **kwargs)
    
    def _process_with_core(self, X: Union[np.ndarray, pd.DataFrame], 
                          y: Union[np.ndarray, pd.Series],
                          method: str, **kwargs) -> Dict[str, Any]:
        """Process using core framework."""
        tprint_info("⚙️ Using core framework")
        
        return core_select_features(X, y, method=method, **kwargs)
    
    def compare_methods(self, X: Union[np.ndarray, pd.DataFrame], 
                       y: Union[np.ndarray, pd.Series],
                       methods: List[str], **kwargs) -> Dict[str, Any]:
        """Compare multiple feature selection methods."""
        if self.parallel_selector:
            tprint_info(f"⚡ Comparing {len(methods)} methods in parallel")
            return self.parallel_selector.parallel_selection(X, y, methods, **kwargs)
        else:
            # Sequential comparison
            results = {}
            for method in methods:
                results[method] = self.select_features(X, y, method, **kwargs)
            return {'success': True, 'results': results}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from all components."""
        summary = {
            'framework': 'EnhancedFeatureSelectionFramework',
            'components': {}
        }
        
        # Collect stats from each component
        if self.cache_manager:
            summary['components']['cache'] = self.cache_manager.get_performance_stats()
        
        if self.memory_selector:
            summary['components']['memory'] = self.memory_selector.get_performance_stats()
        
        if self.parallel_selector:
            summary['components']['parallel'] = self.parallel_selector.get_performance_stats()
        
        if self.vectorized_selector:
            summary['components']['vectorized'] = self.vectorized_selector.get_performance_stats()
        
        if self.sparse_selector:
            summary['components']['sparse'] = self.sparse_selector.get_performance_stats()
        
        if self.chunked_processor:
            summary['components']['chunked'] = self.chunked_processor.get_processing_stats()
        
        if self.error_handler:
            summary['components']['error_handling'] = self.error_handler.get_error_summary()
        
        return summary

# Global enhanced framework instance
_enhanced_framework: Optional[EnhancedFeatureSelectionFramework] = None

def get_enhanced_framework(config: Optional[EnhancedFeatureSelectionConfig] = None) -> EnhancedFeatureSelectionFramework:
    """Get global enhanced framework instance."""
    global _enhanced_framework
    if _enhanced_framework is None:
        _enhanced_framework = EnhancedFeatureSelectionFramework(config)
    return _enhanced_framework

def enhanced_select_features(X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series],
                           method: str = 'comprehensive',
                           config: Optional[EnhancedFeatureSelectionConfig] = None,
                           **kwargs) -> Dict[str, Any]:
    """Enhanced feature selection with automatic optimization."""
    framework = get_enhanced_framework(config)
    return framework.select_features(X, y, method, **kwargs)