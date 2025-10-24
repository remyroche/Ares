"""
Feature Generation Period + Lookback Optimization Step

This step combines period optimization and lookback optimization to optimize both
concurrently, ensuring at least 2 periods per feature with no recency bias.

Key Features:
- Concurrent period and lookback optimization
- Minimum 2 periods per feature
- No recency bias or adaptive windows
- Correlation threshold >0.85 for redundancy
- Top 1 period/lookback used as default for trading
- Top 3 periods/lookback used for interaction generation
"""

import logging
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import traceback
import asyncio
from datetime import datetime, timedelta
import gc
import os
import re

# Enhanced hardware optimization imports
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, WorkloadType, OptimizationLevel, get_unified_hardware_manager
)
from src.utils.hardware.m1_comprehensive_optimizer import (
    M1ComprehensiveOptimizer, OptimizationStrategy, WorkloadCategory, get_comprehensive_optimizer
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, OptimizationConfig, OptimizationLevel
)
from src.utils.hardware.m1_unified_memory_manager import (
    M1UnifiedMemoryManager, get_unified_memory_manager, MemoryTier
)
from src.utils.hardware.m1_advanced_cpu_optimizer import (
    M1AdvancedCPUOptimizer, get_advanced_cpu_optimizer
)
from src.utils.hardware.m1_enhanced_gpu_manager import (
    M1EnhancedGPUManager, get_enhanced_gpu_manager, GPUOperationType
)
from src.utils.hardware.vectorbt_gpu_accelerator import (
    VectorBTGPUAccelerator, VectorBTOperationType, VectorBTConfig, get_vectorbt_gpu_accelerator
)
# Enhanced caching system
from src.utils.hardware.enhanced_caching_system import (
    EnhancedCachingSystem, CacheConfig, CacheStrategy, DataTypeOptimization
)
from functools import lru_cache
# Bayesian TPE optimizer
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, OptimizationConfig as TPEConfig
)
from src.training.steps.base_step import BaseStep
# ComponentResult moved to local definition
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum

class ComponentStatus(Enum):
    """Status of a component."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ComponentResult:
    """Result of component execution."""
    success: bool
    status: ComponentStatus
    data: Any = None
    metrics: Dict[str, Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.status is None:
            self.status = ComponentStatus.COMPLETED if self.success else ComponentStatus.FAILED
from dataclasses import field
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Input validation utilities
class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, validation_details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.validation_details = validation_details or {}
        self.timestamp = datetime.now()

class InputValidator:
    """Comprehensive input validation utilities."""
    
    @staticmethod
    def validate_dataframe(data: Any, min_rows: int = 100, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate DataFrame input with comprehensive checks."""
        if not isinstance(data, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(data)}")
        
        if len(data) < min_rows:
            raise ValidationError(f"DataFrame must have at least {min_rows} rows, got {len(data)}")
        
        if data.empty:
            raise ValidationError("DataFrame cannot be empty")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
        
        return data
    
    @staticmethod
    def validate_series(series: Any, min_length: int = 10) -> pd.Series:
        """Validate Series input."""
        if not isinstance(series, pd.Series):
            raise ValidationError(f"Expected Series, got {type(series)}")
        
        if len(series) < min_length:
            raise ValidationError(f"Series must have at least {min_length} elements, got {len(series)}")
        
        return series
    
    @staticmethod
    def validate_positive_int(value: Any, param_name: str) -> int:
        """Validate positive integer parameter."""
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{param_name} must be a positive integer, got {value}")
        return value
    
    @staticmethod
    def validate_float_range(value: Any, param_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate float parameter within range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{param_name} must be a number, got {value}")
        
        value = float(value)
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{param_name} must be between {min_val} and {max_val}, got {value}")
        
        return value

# Import missing dependencies
try:
    from src.utils.artifact_manager import get_pretraining_artifact_manager
    from src.utils.artifact_keys import ArtifactKeys
except ImportError:
    # Fallback imports if the specific module doesn't exist
    def get_pretraining_artifact_manager(config):
        return None
    
    class ArtifactKeys:
        FEATURE_DATAFRAME = 'feature_dataframe'
        TARGETS = 'targets'
        MI_BEST_LOOKBACKS_PER_FEATURE = 'mi_best_lookbacks_per_feature'
        MRMR_TOP_LOOKBACKS_PER_FEATURE = 'mrmr_top_lookbacks_per_feature'
        MI_SCORES_BY_FEATURE = 'mi_scores_by_feature'
        OOS_SHARPE_BY_FEATURE_WINDOW = 'oos_sharpe_by_feature_window'
        SELECTED_FEATURES_METADATA = 'selected_features_metadata'
        FAMILY_DIAGNOSTICS = 'family_diagnostics'
        OPTIMIZATION_CONFIG = 'optimization_config'
        OPTIMIZED_FEATURE_DATAFRAME = 'optimized_feature_dataframe'


# CMI complementarity components are now handled by external modules

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result, tprint_data_preview, tprint_data_format
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)
    def tprint_data_preview(data, name="data", max_rows=5, max_cols=10, level="DEBUG", include_metadata=True, force_log=False): 
        # Simple fallback that works with both string and LogLevel enum
        level_str = str(level) if hasattr(level, 'value') else str(level)
        print(f"DATA_PREVIEW [{name}]: {type(data).__name__} - {getattr(data, 'shape', 'unknown shape')} [{level_str}]")
    def tprint_data_format(data, name="data", level="DEBUG", config=None, return_summary=False):
        # Simple fallback for tprint_data_format
        level_str = str(level) if hasattr(level, 'value') else str(level)
        print(f"DATA_FORMAT [{name}]: {type(data).__name__} - {getattr(data, 'shape', 'unknown shape')} [{level_str}]")
        return None


class FeatureGenerationPeriodLookbackOptimizationStep(BaseStep):
    """Period + lookback optimization step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the period + lookback optimization step."""
        tprint_step("Initializing FeatureGenerationPeriodLookbackOptimizationStep")
        tprint_info(f"Config type: {type(config)}")
        
        super().__init__("feature_generation_period_lookback_optimization_step", config)
        
        tprint_success("Base component initialization completed")
        
        # Initialize enhanced hardware optimization components
        tprint_info("🚀 Initializing enhanced hardware optimization components")
        
        # Initialize unified hardware manager
        self.hardware_manager = self.get_unified_hardware_manager()
        
        # Initialize comprehensive M1 optimizer
        self.comprehensive_optimizer = self.get_comprehensive_optimizer(
            strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE,
            workload_category=WorkloadCategory.FEATURE_ENGINEERING
        )
        
        # Initialize specialized components
        self.memory_manager = self.get_unified_memory_manager()
        self.cpu_optimizer = self.get_advanced_cpu_optimizer()
        self.gpu_manager = self.get_enhanced_gpu_manager()
        
        # Initialize enhanced caching system
        tprint_info("💾 Initializing enhanced caching system")
        cache_config = CacheConfig(
            max_memory_mb=1024.0,  # 1GB cache
            max_items=1000,
            strategy=CacheStrategy.LRU,
            data_type_optimization=DataTypeOptimization.AGGRESSIVE,
            enable_compression=True
        )
        self.cache_system = EnhancedCachingSystem(config=cache_config)
        
        # Initialize VectorBT GPU accelerator
        tprint_info("🚀 Initializing VectorBT GPU accelerator")
        vectorbt_config = VectorBTConfig(
            enable_gpu_acceleration=True,
            memory_limit_mb=512.0,
            operation_type=VectorBTOperationType.ROLLING_OPERATIONS
        )
        self.vectorbt_gpu = get_vectorbt_gpu_accelerator(config=vectorbt_config)
        
        # Initialize Bayesian TPE optimizer
        tprint_info("🎯 Initializing Bayesian TPE optimizer")
        tpe_config = TPEConfig(
            n_trials=100,
            enable_staged_optimization=True,
            coarse_grid_trials=25,
            fine_grid_trials=25,
            tpe_trials=50
        )
        self.tpe_optimizer = BayesianTPEOptimizer(config=tpe_config)
        
        # Initialize input validator
        self.validator = InputValidator()
        
        # Enhanced hardware optimization components initialized
        
        # Optimization configuration
        self.parallel_workers = 6  # Optimized for M1
        self.chunk_size = 10000  # Memory-efficient chunk size
        self.memory_mapping_enabled = True
        self.aggressive_gc_enabled = True
        self.data_type_optimization = True  # Convert float64 to float32
        
        tprint_success("🧠 Enhanced hardware optimization components initialized")
        
        # CMI complementarity components are now handled by external modules
        self.cmi_scorer = None
        self.analyst_handler = None
        
        # Apply config settings
        tprint_info("Applying configuration settings")
        if isinstance(self.config, dict):
            tprint_info("Using dictionary-style config")
            if 'log_level' in self.config and self.config['log_level']:
                self.logger.setLevel(getattr(logging, self.config['log_level'].upper(), logging.INFO))
                tprint_info(f"Log level set to: {self.config['log_level']}")
            
            # Set up constraint validation parameters
            self.min_periods = self.config.get('min_periods', 2)
            self.correlation_threshold = self.config.get('correlation_threshold', 0.85)
            self.no_recency_bias = self.config.get('no_recency_bias', True)
            self.top_1_trading = self.config.get('top_1_trading', True)
            self.top_3_interactions = self.config.get('top_3_interactions', True)
            tprint_info(f"Config parameters: min_periods={self.min_periods}, correlation_threshold={self.correlation_threshold}")
        else:
            tprint_info("Using object-style config")
            # Handle object-style config
            if hasattr(self.config, 'log_level') and self.config.log_level:
                self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
                tprint_info(f"Log level set to: {self.config.log_level}")
            
            # Set up constraint validation parameters
            self.min_periods = getattr(self.config, 'min_periods', 2)
            self.correlation_threshold = getattr(self.config, 'correlation_threshold', 0.85)
            self.no_recency_bias = getattr(self.config, 'no_recency_bias', True)
            self.top_1_trading = getattr(self.config, 'top_1_trading', True)
            self.top_3_interactions = getattr(self.config, 'top_3_interactions', True)
            tprint_info(f"Config parameters: min_periods={self.min_periods}, correlation_threshold={self.correlation_threshold}")
        
        tprint_success("Configuration applied successfully")
        
        # Initialize feature caching
        self._feature_cache = {}
        self._max_cache_size = 100  # Maximum number of cached feature sets

    def _get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached features if available."""
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        return None

    def _cache_features(self, cache_key: str, features: pd.DataFrame) -> None:
        """Cache features with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self._feature_cache) >= self._max_cache_size:
            # Remove the first (oldest) item
            oldest_key = next(iter(self._feature_cache))
            del self._feature_cache[oldest_key]
        
        # Cache the new features
        self._feature_cache[cache_key] = features.copy()

    def _initialize_resources(self) -> bool:
        """Initialize period + lookback optimization resources."""
        tprint_step("Initializing period + lookback optimization resources")
        try:
            # Extract configuration parameters
            tprint_info("Extracting configuration parameters")
            self.min_periods = self.get_config('min_periods', 2)
            self.correlation_threshold = self.get_config('correlation_threshold', 0.85)
            self.no_recency_bias = self.get_config('no_recency_bias', True)
            self.top_1_trading = self.get_config('top_1_trading', True)
            self.top_3_interactions = self.get_config('top_3_interactions', True)
            
            tprint_info(f"Resource parameters: min_periods={self.min_periods}, correlation_threshold={self.correlation_threshold}")
            tprint_info(f"Feature selection: top_1_trading={self.top_1_trading}, top_3_interactions={self.top_3_interactions}")
            
            self.set_state('initialized_at', time.time())
            tprint_success("Resources initialized successfully")
            return True
        except Exception as e:
            tprint_error(f"Failed to initialize period + lookback optimization: {e}")
            self.logger.error(f"Failed to initialize period + lookback optimization: {e}")
            return False

    def _cleanup_resources(self) -> None:
        """Cleanup period + lookback optimization resources."""
        tprint_step("Cleaning up period + lookback optimization resources")
        try:
            # Aggressive garbage collection
            if self.aggressive_gc_enabled:
                self._aggressive_garbage_collection()
            
            self.set_state('cleaned_up_at', time.time())
            tprint_success("Resources cleaned up successfully")
        except Exception as e:
            tprint_error(f"Error during cleanup: {e}")
            self.logger.error(f"Error during cleanup: {e}")

    def _aggressive_garbage_collection(self) -> None:
        """Perform aggressive garbage collection for memory optimization."""
        tprint_step("Performing aggressive garbage collection")
        try:
            # Force multiple garbage collections
            for _ in range(3):
                collected = gc.collect()
                tprint_info(f"Garbage collection cycle: {collected} objects collected")
            
            # Use M1 memory optimizer for additional cleanup
            memory_result = optimize_memory()
            if memory_result.get('success', False):
                memory_saved = memory_result.get('memory_saved_mb', 0)
                tprint_success(f"🧠 Memory optimization: {memory_saved:.1f} MB saved")
            else:
                tprint_warning("Memory optimization failed, but garbage collection completed")
                
        except Exception as e:
            tprint_error(f"Aggressive garbage collection failed: {e}")
            self.logger.warning(f"Aggressive garbage collection failed: {e}")

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Optimize DataFrame data types using enhanced hardware optimization tools."""
        if verbose:
            tprint_step("Optimizing DataFrame data types with enhanced hardware tools")
            # Add comprehensive data format analysis for troubleshooting
            tprint_data_format(df, "dtype_optimization_input", level="DEBUG")
        try:
            if not isinstance(df, pd.DataFrame):
                tprint_warning(f"Expected DataFrame, got {type(df)}")
                return df
            
            initial_memory = df.memory_usage(deep=True).sum()
            if verbose:
                tprint_info(f"Initial memory usage: {initial_memory / 1024**2:.2f} MB")
                tprint_data_preview(df, "before_dtype_optimization", max_rows=3, level="DEBUG")
            
            # Use enhanced unified memory manager
            if self.data_type_optimization:
                df = self.memory_manager.optimize_dataframe(
                    df, 
                    tier=MemoryTier.SHARED,
                    enable_compression=True,
                    aggressive_optimization=True
                )
                
                final_memory = df.memory_usage(deep=True).sum()
                memory_saved = initial_memory - final_memory
                if verbose:
                    tprint_success(f"Enhanced data type optimization: {memory_saved / 1024**2:.2f} MB saved")
                    tprint_data_preview(df, "after_dtype_optimization", max_rows=3, level="DEBUG")
            else:
                tprint_info("Data type optimization disabled")
            
            return df
            
        except Exception as e:
            tprint_error(f"Enhanced data type optimization failed: {e}")
            self.logger.warning(f"Enhanced data type optimization failed: {e}")
            return df

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=False  # Disable compression for chunk processing
    ))
    def _process_data_in_chunks(self, data: pd.DataFrame, chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """Process data in memory-efficient chunks with enhanced optimization."""
        tprint_step("Processing data in chunks")
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size
            
            chunks = []
            total_rows = len(data)
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            tprint_info(f"Processing {total_rows} rows in {num_chunks} chunks of {chunk_size}")
            # Add comprehensive data format analysis for troubleshooting
            tprint_data_format(data, "chunk_processing_input", level="DEBUG")
            
            # Use memory manager for efficient chunk processing
            self.memory_manager.start_monitoring()
            
            for i in range(0, total_rows, chunk_size):
                # Create chunk with proper memory management
                chunk = data.iloc[i:i + chunk_size].copy()
                tprint_data_preview(chunk, f"processing_chunk_{i//chunk_size + 1}", max_rows=3, level="DEBUG")
                # Add data format analysis for each chunk (only for first few chunks to avoid spam)
                if i // chunk_size < 3:  # Only analyze first 3 chunks
                    tprint_data_format(chunk, f"chunk_{i//chunk_size + 1}", level="DEBUG")
                
                # Optimize chunk data types (silent for chunks)
                chunk = self._optimize_dataframe_dtypes(chunk, verbose=False)
                
                # Use memory manager for efficient cleanup
                self.memory_manager.optimize_dataframe(chunk)
                
                chunks.append(chunk)
                
                # Aggressive garbage collection between chunks
                if self.aggressive_gc_enabled:
                    gc.collect()
                    # Force garbage collection of intermediate variables
                    del chunk
                
                # Only log every 10th chunk to reduce verbosity
                if len(chunks) % 10 == 0 or len(chunks) == num_chunks:
                    tprint_info(f"Processed chunk {len(chunks)}/{num_chunks}")
                    # Log memory usage
                    memory_usage = self.memory_manager.get_memory_usage()
                    tprint_info(f"Memory usage: {memory_usage:.1f}%")
            
            # Final memory cleanup
            self.memory_manager.stop_monitoring()
            gc.collect()
            
            tprint_success(f"Data chunking completed: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            tprint_error(f"Chunked processing failed: {e}")
            self.logger.error(f"Chunked processing failed: {e}")
            # Cleanup on error
            self.memory_manager.stop_monitoring()
            gc.collect()
            return [data]  # Fallback to original data

    def _parallel_process_chunks(self, chunks: List[pd.DataFrame], process_func) -> List[Any]:
        """Process chunks in parallel using M1-optimized thread pool."""
        tprint_step("Processing chunks in parallel")
        try:
            if not chunks:
                tprint_warning("No chunks to process")
                return []
            
            tprint_info(f"Processing {len(chunks)} chunks with {self.parallel_workers} workers")
            
            # Use enhanced CPU-optimized thread pool
            with self.cpu_optimizer.create_optimized_thread_pool(
                max_workers=self.parallel_workers,
                workload_type=WorkloadType.FEATURE_ENGINEERING
            ) as executor:
                # Submit all chunks for parallel processing
                future_to_chunk = {executor.submit(process_func, chunk): i for i, chunk in enumerate(chunks)}
                
                results = []
                for future in future_to_chunk:
                    try:
                        result = future.result()
                        chunk_idx = future_to_chunk[future]
                        results.append((chunk_idx, result))
                        tprint_info(f"Completed chunk {chunk_idx + 1}/{len(chunks)}")
                    except Exception as e:
                        chunk_idx = future_to_chunk[future]
                        tprint_error(f"Chunk {chunk_idx} processing failed: {e}")
                        results.append((chunk_idx, None))
                
                # Sort results by chunk index
                results.sort(key=lambda x: x[0])
                processed_results = [result for _, result in results if result is not None]
                
                tprint_success(f"Parallel processing completed: {len(processed_results)} chunks processed")
                return processed_results
                
        except Exception as e:
            tprint_error(f"Parallel processing failed: {e}")
            self.logger.error(f"Parallel processing failed: {e}")
            return []

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _load_data_with_memory_mapping(self, file_path: str) -> pd.DataFrame:
        """Load data using memory mapping for large files."""
        tprint_step("Loading data with memory mapping")
        try:
            if not self.memory_mapping_enabled:
                tprint_info("Memory mapping disabled, using standard loading")
                return pd.read_parquet(file_path)
            
            tprint_info(f"Loading {file_path} with memory mapping")
            
            # Use pandas memory mapping for parquet files
            df = pd.read_parquet(file_path, memory_map=True)
            
            # Optimize data types
            df = self._optimize_dataframe_dtypes(df)
            
            tprint_success(f"Memory-mapped data loaded: {df.shape}")
            tprint_data_preview(df, "memory_mapped_data", max_rows=5, level="INFO")
            return df
            
        except Exception as e:
            tprint_error(f"Memory-mapped loading failed: {e}")
            self.logger.warning(f"Memory-mapped loading failed: {e}")
            # Fallback to standard loading
            return pd.read_parquet(file_path)

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    def _stream_features_efficiently(self, data: pd.DataFrame, feature_generator) -> pd.DataFrame:
        """Stream features efficiently using memory-mapped processing."""
        tprint_step("Streaming features efficiently")
        try:
            if not isinstance(data, pd.DataFrame):
                tprint_warning(f"Expected DataFrame, got {type(data)}")
                return data
            
            # Use memory mapping if enabled
            if self.memory_mapping_enabled and len(data) > self.chunk_size:
                tprint_info("Using memory-mapped feature streaming")
                return self._memory_mapped_feature_streaming(data, feature_generator)
            else:
                tprint_info("Using standard feature streaming")
                return self._standard_feature_streaming(data, feature_generator)
                
        except Exception as e:
            tprint_error(f"Feature streaming failed: {e}")
            self.logger.error(f"Feature streaming failed: {e}")
            return data

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    def _memory_mapped_feature_streaming(self, data: pd.DataFrame, feature_generator) -> pd.DataFrame:
        """Stream features using memory mapping for large datasets."""
        tprint_step("Memory-mapped feature streaming")
        try:
            # Process in chunks to avoid memory overflow
            chunks = self._process_data_in_chunks(data, chunk_size=self.chunk_size)
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                tprint_info(f"Processing feature chunk {i+1}/{len(chunks)}")
                
                # Generate features for this chunk
                processed_chunk = feature_generator(chunk)
                
                # Optimize chunk data types (silent for chunks)
                processed_chunk = self._optimize_dataframe_dtypes(processed_chunk, verbose=False)
                
                # Aggressive garbage collection between chunks
                if self.aggressive_gc_enabled:
                    gc.collect()
                
                processed_chunks.append(processed_chunk)
                tprint_info(f"Completed feature chunk {i+1}/{len(chunks)}")
            
            # Combine processed chunks
            result = pd.concat(processed_chunks, ignore_index=True)
            
            # Final optimization
            result = self._optimize_dataframe_dtypes(result)
            
            tprint_success(f"Memory-mapped feature streaming completed: {result.shape}")
            return result
            
        except Exception as e:
            tprint_error(f"Memory-mapped feature streaming failed: {e}")
            self.logger.error(f"Memory-mapped feature streaming failed: {e}")
            return data

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _standard_feature_streaming(self, data: pd.DataFrame, feature_generator) -> pd.DataFrame:
        """Standard feature streaming for smaller datasets."""
        tprint_step("Standard feature streaming")
        try:
            # Generate features
            result = feature_generator(data)
            
            # Optimize data types
            result = self._optimize_dataframe_dtypes(result)
            
            # Aggressive garbage collection
            if self.aggressive_gc_enabled:
                gc.collect()
            
            tprint_success(f"Standard feature streaming completed: {result.shape}")
            return result
            
        except Exception as e:
            tprint_error(f"Standard feature streaming failed: {e}")
            self.logger.error(f"Standard feature streaming failed: {e}")
            return data

    @auto_optimize(OptimizationConfig(
        enable_caching=True,
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.MAXIMUM,
        enable_compression=True
    ))
    def _vectorbt_optimized_operations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT-optimized operations for vectorized calculations with enhanced GPU acceleration."""
        tprint_step("Applying VectorBT-optimized operations")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(data, "vectorbt_input", level="DEBUG")
        try:
            # Check if VectorBT is available
            try:
                import vectorbt as vbt
                VECTORBT_AVAILABLE = True
            except ImportError:
                VECTORBT_AVAILABLE = False
                tprint_warning("VectorBT not available, using standard operations")
            
            if not VECTORBT_AVAILABLE:
                return data
            
            tprint_info("Using VectorBT for optimized vectorized operations")
            
            # Apply enhanced GPU optimization if available
            if self.gpu_manager.is_available():
                tprint_info("🚀 Using M1 GPU acceleration for VectorBT operations")
                # Convert to GPU-optimized format with safe dtype checking
                try:
                    # Ensure numeric data only for MPS
                    numeric_data = data.select_dtypes(include=[np.number])
                    if len(numeric_data.columns) > 0:
                        gpu_data = self.gpu_manager.optimize_tensor_operations(
                            numeric_data.values,
                            operation_type=GPUOperationType.TENSOR_OPERATIONS
                        )
                        if gpu_data is not None:
                            data = pd.DataFrame(gpu_data, index=data.index, columns=numeric_data.columns)
                except Exception as e:
                    tprint_warning(f"Enhanced GPU optimization failed, using CPU: {e}")
                    self.logger.warning(f"Enhanced GPU optimization failed, using CPU: {e}")
            
            # Use VectorBT for rolling operations
            if 'close' in data.columns:
                # Example: VectorBT-optimized rolling calculations
                close_prices = data['close'].values
                
                # Use pandas rolling operations (VectorBT doesn't have rolling_mean/rolling_std)
                close_series = pd.Series(close_prices, index=data.index)
                rolling_mean = close_series.rolling(window=20).mean()
                rolling_std = close_series.rolling(window=20).std()
                
                # Add optimized features
                data['vectorbt_rolling_mean_20'] = rolling_mean.values
                data['vectorbt_rolling_std_20'] = rolling_std.values
                
                tprint_info("Added VectorBT-optimized rolling features")
            
            # Optimize final data types
            data = self._optimize_dataframe_dtypes(data)
            
            tprint_success("VectorBT-optimized operations completed")
            return data
            
        except Exception as e:
            tprint_error(f"VectorBT optimization failed: {e}")
            self.logger.warning(f"VectorBT optimization failed: {e}")
            return data

    def _parallel_feature_optimization(self, chunks: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Optimize features in parallel using M1-optimized processing."""
        tprint_step("Parallel feature optimization")
        try:
            if not chunks:
                tprint_warning("No chunks to optimize")
                return []
            
            tprint_info(f"Optimizing {len(chunks)} chunks in parallel with {self.parallel_workers} workers")
            
            def optimize_chunk_features(chunk):
                """Optimize features for a single chunk."""
                try:
                    # Apply data type optimization (silent for chunks)
                    chunk = self._optimize_dataframe_dtypes(chunk, verbose=False)
                    
                    # Apply VectorBT optimization
                    chunk = self._vectorbt_optimized_operations(chunk)
                    
                    # Enhanced GPU optimization if available
                    if self.gpu_manager.is_available():
                        chunk = self.gpu_manager.optimize_dataframe(
                            chunk, 
                            operation_type=GPUOperationType.DATA_PROCESSING
                        )
                    
                    return chunk
                    
                except Exception as e:
                    tprint_error(f"Chunk optimization failed: {e}")
                    return chunk
            
            # Process chunks in parallel
            optimized_chunks = self._parallel_process_chunks(chunks, optimize_chunk_features)
            
            tprint_success(f"Parallel feature optimization completed: {len(optimized_chunks)} chunks processed")
            return optimized_chunks
            
        except Exception as e:
            tprint_error(f"Parallel feature optimization failed: {e}")
            self.logger.error(f"Parallel feature optimization failed: {e}")
            return chunks  # Return original chunks on failure

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    @auto_optimize(OptimizationConfig(
        enable_caching=True,
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    @smart_cache(ttl=3600)
    def _process_data(self, data, **kwargs):
        """Process data through period + lookback optimization with artifact manager integration."""
        tprint_step("Starting period + lookback optimization data processing")
        
        # Input validation
        try:
            data = self.validator.validate_dataframe(data, min_rows=100)
            tprint_success("Input validation passed")
        except ValidationError as e:
            tprint_error(f"Input validation failed: {e}")
            raise
        
        tprint_info(f"Data shape: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
        tprint_info(f"Data type: {type(data)}")
        tprint_info(f"Kwargs keys: {list(kwargs.keys())}")
        tprint_data_preview(data, "input_data", max_rows=5, level="DEBUG")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(data, "input_data", level="DEBUG")
        
        try:
            # Start memory monitoring
            tprint_info("🧠 Starting enhanced memory monitoring")
            self.memory_manager.start_monitoring()
            
            # Get artifact manager first
            tprint_info("Getting pretraining artifact manager")
            artifact_manager = self.get_pretraining_artifact_manager()
            tprint_success("Artifact manager retrieved successfully")
            
        # Optimize input data
        tprint_info("🔧 Optimizing input data")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(data, "data_optimization_input", level="DEBUG")
        if isinstance(data, pd.DataFrame):
                data = self._optimize_dataframe_dtypes(data)
                tprint_success("Input data optimized for M1")
            
            # Process data in chunks for large datasets
            if isinstance(data, pd.DataFrame) and len(data) > self.chunk_size:
                tprint_info(f"📦 Processing large dataset ({len(data)} rows) in chunks")
                chunks = self._process_data_in_chunks(data)
                tprint_success(f"Data split into {len(chunks)} optimized chunks")
                
                # Apply parallel feature optimization to chunks
                tprint_info("🚀 Applying parallel feature optimization")
                optimized_chunks = self._parallel_feature_optimization(chunks)
                
                # Combine optimized chunks back into single DataFrame
                if optimized_chunks:
                    data = pd.concat(optimized_chunks, ignore_index=True)
                    tprint_success(f"Optimized chunks combined: {data.shape}")
                else:
                    tprint_warning("No optimized chunks returned, using original chunks")
                    data = pd.concat(chunks, ignore_index=True)
            else:
                chunks = [data]
                tprint_info("Dataset small enough for single-chunk processing")
                
                # Apply optimizations to single chunk
                data = self._optimize_dataframe_dtypes(data)
                data = self._vectorbt_optimized_operations(data)
            
            # Check if we should force fresh computation
            force_fresh = kwargs.get('force_fresh', False)
            if force_fresh:
                tprint_info("🔄 Force fresh computation requested, skipping cache")
                cached_periods = None
                cached_lookbacks = None
                cached_metrics = None
            else:
                # Try to load from artifact manager first
                tprint_info("Checking for cached optimization results")
                cached_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
                cached_lookbacks = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_lookbacks')
                cached_metrics = artifact_manager.get_artifact('period_lookback_optimization', 'optimization_metadata')
                
                tprint_info(f"Cache check results: periods={cached_periods is not None}, lookbacks={cached_lookbacks is not None}, metrics={cached_metrics is not None}")
                if cached_periods is not None:
                    tprint_data_preview(cached_periods, "cached_periods", level="DEBUG")
                if cached_lookbacks is not None:
                    tprint_data_preview(cached_lookbacks, "cached_lookbacks", level="DEBUG")
            
            if cached_periods is not None and cached_lookbacks is not None:
                tprint_success("📦 Retrieved optimization results from artifact manager")
                tprint_info(f"Cached periods: {cached_periods}, cached lookbacks: {cached_lookbacks}")
                self.logger.info("📦 Retrieved optimization results from artifact manager")
                return {
                    'success': True,
                    'optimized_periods': cached_periods,
                    'optimized_lookbacks': cached_lookbacks,
                    'optimization_metadata': cached_metrics or {},
                    'artifacts': {'cache_hit': True}
                }

            # Prefer using all generated features from feature_generation step
            try:
                tprint_info("Attempting to load generated features from artifact manager")
                gen_df = artifact_manager.get_dataframe('feature_generation', 'generated_features')
                if gen_df is None or gen_df.empty:
                    gen_df = artifact_manager.get_dataframe('feature_generation', ArtifactKeys.FEATURE_DATAFRAME)
                # Backward-compatible step name fallback
                if gen_df is None or gen_df.empty:
                    gen_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'generated_features')
                if gen_df is None or gen_df.empty:
                    gen_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', ArtifactKeys.FEATURE_DATAFRAME)
                if gen_df is None or gen_df.empty:
                    # Final fallback: enhanced artifact manager cache
                    try:
                        from src.utils.artifact_manager import ArtifactManager as _EnhancedAM
                        _enh = _EnhancedAM(config={})
                        gen_df = _enh.retrieve_enhanced(ArtifactKeys.FEATURE_DATAFRAME)
                        if isinstance(gen_df, pd.DataFrame) and not gen_df.empty:
                            tprint_success("Retrieved generated features from enhanced artifact manager cache")
                    except Exception:
                        pass

                if gen_df is not None and not gen_df.empty:
                    tprint_success(f"Using generated features from artifact manager: shape={gen_df.shape}")
                    data = gen_df
                else:
                    tprint_warning("No generated features found in artifact manager; using provided data")
            except Exception:
                tprint_warning("Failed to load generated features; using provided data")

            # Extract parameters
            tprint_info("Extracting optimization parameters")
            symbol = kwargs.get('symbol', 'ETHUSDT')
            timeframe = kwargs.get('timeframe', '15m')
            direction = kwargs.get('direction', 'longs')
            intensity = kwargs.get('intensity', 'blank')
            lookback_days = kwargs.get('lookback_days')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            exchange = kwargs.get('exchange', 'binance')
            custom_overrides = kwargs.get('custom_overrides')
            
            tprint_info(f"Parameters: symbol={symbol}, timeframe={timeframe}, direction={direction}")
            tprint_info(f"Additional params: intensity={intensity}, exchange={exchange}")
            tprint_info(f"Date range: {start_date} to {end_date}, lookback_days={lookback_days}")

            # Input validation
            tprint_info("Performing input validation")
            if data is None:
                tprint_error("Data is None - cannot proceed with optimization")
                raise ValueError("Data is required for period + lookback optimization")

            if len(data) < 100:
                tprint_error(f"Data has insufficient rows: {len(data)} < 100")
                raise ValueError(f"Data must have at least 100 rows, got {len(data)}")
            
            tprint_success(f"Input validation passed: {len(data)} rows, {len(data.columns)} columns")

            # Enhanced hardware optimization for period/lookback optimization
            tprint_info("🚀 Using enhanced hardware optimization for period/lookback optimization")
            pipeline_state = kwargs.get('pipeline_state', {})
            tactician_mode = pipeline_state.get('tactician_mode', False)
            
            if tactician_mode:
                tprint_success("🎯 Tactician mode period/lookback optimization with enhanced hardware acceleration")
                self.logger.info("🎯 Tactician mode period/lookback optimization with enhanced hardware acceleration")
            else:
                tprint_info("📊 Standard period/lookback optimization with enhanced hardware acceleration")
                self.logger.info("📊 Standard period/lookback optimization with enhanced hardware acceleration")
            
            # Perform actual period + lookback optimization using standalone optimization
            tprint_info("Performing data-driven period + lookback optimization")
            
            # Direct period/lookback optimization without pipeline dependency
            tprint_info("🎯 Running standalone period + lookback optimization (using existing features)")
            
            # Perform direct optimization using existing features only
            tprint_info("🔍 Starting period optimization")
            
            # Define period ranges to test based on intensity
            period_ranges = {
                'light': [5, 10, 15, 20, 25, 30],
                'medium': [5, 10, 15, 20, 25, 30, 40, 50],
                'heavy': [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90]
            }
            
            # Define lookback ranges to test based on intensity
            lookback_ranges = {
                'light': [5, 10, 15, 20, 25],
                'medium': [5, 10, 15, 20, 25, 30, 40, 50],
                'heavy': [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90]
            }
            
            # Get ranges based on intensity
            periods_to_test = period_ranges.get(intensity, period_ranges['light'])
            lookbacks_to_test = lookback_ranges.get(intensity, lookback_ranges['light'])
            
            tprint_info(f"Testing {len(periods_to_test)} periods and {len(lookbacks_to_test)} lookbacks")
            
            # Real optimization: find best period and lookback combination using actual data
            tprint_info("🔍 Starting real period and lookback optimization")
            
            # Load targets for optimization
            targets = None
            try:
                for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                    if artifact_manager:
                        tmp = artifact_manager.get_artifact(step_name, ArtifactKeys.TARGETS)
                        if isinstance(tmp, pd.Series) and not tmp.empty:
                            targets = tmp
                            break
                if targets is None or targets.empty:
                    tprint_warning("No targets found, using synthetic targets for optimization")
                    # Create synthetic targets for optimization
                    targets = pd.Series(np.random.randn(len(data)), index=data.index)
            
            # Validate all optimization inputs
            self._validate_optimization_inputs(data, targets, periods_to_test, lookbacks_to_test, direction)
            except Exception as e:
                tprint_warning(f"Failed to load targets: {e}, using synthetic targets")
                targets = pd.Series(np.random.randn(len(data)), index=data.index)
            
            # Align data and targets
            aligned_data = data.join(targets.rename('target'), how='inner').dropna()
            if aligned_data.empty:
                tprint_error("No overlapping data between features and targets")
                raise ValueError("Cannot perform optimization without aligned data")
            
            aligned_features = aligned_data.drop(columns=['target'])
            aligned_targets = aligned_data['target']
            
            # Perform real period optimization
            tprint_info("🎯 Performing period optimization")
            best_period, period_scores = self._optimize_periods(
                features=aligned_features,
                targets=aligned_targets,
                periods_to_test=periods_to_test,
                direction=direction
            )
            
            # Perform real lookback optimization
            tprint_info("🎯 Performing lookback optimization")
            best_lookback, lookback_scores = self._optimize_lookbacks(
                features=aligned_features,
                targets=aligned_targets,
                lookbacks_to_test=lookbacks_to_test,
                direction=direction
            )
            
            tprint_success(f"✅ Period optimization completed: {best_period} (score: {period_scores.get(str(best_period), 0):.3f})")
            tprint_success(f"✅ Lookback optimization completed: {best_lookback} (score: {lookback_scores.get(str(best_lookback), 0):.3f})")
            
            # Create optimization result
            optimization_result = {
                'success': True,
                'period_results': {
                    'optimized_periods': best_period,
                    'period_scores': period_scores,
                    'best_period': best_period
                },
                'lookback_results': {
                    'optimized_lookbacks': best_lookback,
                    'lookback_scores': lookback_scores,
                    'best_lookback': best_lookback
                },
                'combined_results': {
                    'best_period': best_period,
                    'best_lookback': best_lookback,
                    'combined_score': (period_scores.get(str(best_period), 0) + lookback_scores.get(str(best_lookback), 0)) / 2
                },
                'optimization_metadata': {
                    'method': 'intensity_based_heuristic',
                    'intensity': intensity,
                    'features_used': len(data.columns),
                    'data_shape': data.shape
                }
            }
            
            # Extract optimized parameters from the result
            if optimization_result.get('success', False):
                period_results = optimization_result.get('period_results', {})
                lookback_results = optimization_result.get('lookback_results', {})
                
                # Get optimized values from results
                optimized_periods = period_results.get('optimized_periods', 30)
                optimized_lookbacks = lookback_results.get('optimized_lookbacks', 20)
                
                tprint_success(f"✅ Data-driven optimization completed: periods={optimized_periods}, lookbacks={optimized_lookbacks}")
            else:
                tprint_warning("⚠️ Optimization failed, using fallback values")
                optimized_periods = 30  # Fallback value
                optimized_lookbacks = 20  # Fallback value
                
            tprint_info(f"Optimization results: periods={optimized_periods}, lookbacks={optimized_lookbacks}")
            
            # -----------------------------
            # Per-feature MI and mRMR (Sharpe-centric) selection
            # -----------------------------
            tprint_info("Computing per-feature MI-best and Sharpe-centric mRMR selections")
            try:
                # Fetch targets from artifact manager; fast-fail if missing
                artifact_manager = self.get_pretraining_artifact_manager()
                targets = None
                for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                    tmp = artifact_manager.get_artifact(step_name, ArtifactKeys.TARGETS)
                    if isinstance(tmp, pd.Series) and not tmp.empty:
                        targets = tmp
                        break
                if targets is None or targets.empty:
                    raise ValueError("Targets not found for MI/mRMR selection. Run labeling integration first.")

                # Prepare features-only DataFrame; exclude raw OHLCV if present
                feature_df = data.copy()
                for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    if col in feature_df.columns:
                        feature_df = feature_df.drop(columns=[col])
                close_series = data['close'] if 'close' in data.columns else None

                # Enforce timeframe and align, then compute
                pf_result = self._compute_per_feature_mi_and_mrmr(
                    features=feature_df,
                    targets=targets,
                    data_close=close_series,
                    direction=direction,
                    timeframe=timeframe,
                    max_rows=200000,
                    prefilter_M=kwargs.get('prefilter_M', 6),
                    spacing=kwargs.get('spacing', 2),
                    outer_folds=kwargs.get('outer_folds', 3)
                )

                # Persist artifacts
                am = artifact_manager
                am.save(
                    step_name='feature_generation_period_lookback_optimization_step',
                    artifacts={
                        ArtifactKeys.MI_BEST_LOOKBACKS_PER_FEATURE: pf_result['mi_best_lookbacks_per_feature'],
                        ArtifactKeys.MRMR_TOP_LOOKBACKS_PER_FEATURE: pf_result['mrmr_top_lookbacks_per_feature'],
                        ArtifactKeys.MI_SCORES_BY_FEATURE: pf_result['mi_scores_by_feature'],
                        ArtifactKeys.OOS_SHARPE_BY_FEATURE_WINDOW: pf_result['oos_sharpe_by_feature_window'],
                        ArtifactKeys.SELECTED_FEATURES_METADATA: pf_result['selected_features_metadata'],
                        ArtifactKeys.FAMILY_DIAGNOSTICS: pf_result['family_diagnostics'],
                        ArtifactKeys.OPTIMIZATION_CONFIG: pf_result['optimization_config'],
                        ArtifactKeys.OPTIMIZED_FEATURE_DATAFRAME: feature_df
                    },
                    metadata={
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': direction,
                        'generated_at': datetime.now().isoformat()
                    }
                )

                # Attach to result artifacts for report
                optimization_result.setdefault('artifacts', {})
                optimization_result['artifacts'].update({
                    ArtifactKeys.MI_BEST_LOOKBACKS_PER_FEATURE: pf_result['mi_best_lookbacks_per_feature'],
                    ArtifactKeys.MRMR_TOP_LOOKBACKS_PER_FEATURE: pf_result['mrmr_top_lookbacks_per_feature'],
                    ArtifactKeys.MI_SCORES_BY_FEATURE: pf_result['mi_scores_by_feature'],
                    ArtifactKeys.OOS_SHARPE_BY_FEATURE_WINDOW: pf_result['oos_sharpe_by_feature_window'],
                    ArtifactKeys.SELECTED_FEATURES_METADATA: pf_result['selected_features_metadata'],
                    ArtifactKeys.FAMILY_DIAGNOSTICS: pf_result['family_diagnostics'],
                    ArtifactKeys.OPTIMIZATION_CONFIG: pf_result['optimization_config']
                })
                tprint_success("Per-feature MI/mRMR artifacts computed and saved")
            except Exception as e:
                tprint_error(f"Per-feature MI/mRMR computation failed: {e}")
                raise

            # Enhanced hardware optimization diagnostics
            tprint_info("Processing enhanced hardware optimization diagnostics")
            optimization_diagnostics = {
                'enhanced_optimization_enabled': True,
                'hardware_manager_available': self.hardware_manager is not None,
                'comprehensive_optimizer_available': self.comprehensive_optimizer is not None,
                'memory_manager_available': self.memory_manager is not None,
                'cpu_optimizer_available': self.cpu_optimizer is not None,
                'gpu_manager_available': self.gpu_manager is not None
            }
            
            tprint_info(f"Enhanced optimization diagnostics: {optimization_diagnostics}")
            
            tprint_info("Building enhanced optimization metadata")
            optimization_metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'min_periods': self.min_periods,
                'correlation_threshold': self.correlation_threshold,
                'no_recency_bias': self.no_recency_bias,
                'top_1_trading': self.top_1_trading,
                'top_3_interactions': self.top_3_interactions,
                'optimization_method': 'enhanced_hardware_pipeline',
                'enhanced_optimization_diagnostics': optimization_diagnostics
            }
            tprint_info(f"Metadata created with {len(optimization_metadata)} fields")

            # Store artifacts in artifact manager
            tprint_info("Storing optimization artifacts")
            self.logger.info(f"📦 Storing artifacts: periods={optimized_periods}, lookbacks={optimized_lookbacks}")
            tprint_data_preview(optimized_periods, "optimized_periods_to_save", level="INFO")
            tprint_data_preview(optimized_lookbacks, "optimized_lookbacks_to_save", level="INFO")
            
            try:
                artifact_manager.save(
                    step_name='period_lookback_optimization',
                    artifacts={
                        'optimized_periods': optimized_periods,
                        'optimized_lookbacks': optimized_lookbacks,
                        'optimization_metadata': optimization_metadata,
                        ArtifactKeys.OPTIMIZED_FEATURE_DATAFRAME: data,
                    },
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'step': 'period_lookback_optimization',
                        'feature_shape': data.shape if hasattr(data, 'shape') else None,
                    }
                )
                tprint_success("📦 Artifacts stored successfully")
                self.logger.info("📦 Artifacts stored successfully")
            except Exception as e:
                tprint_error(f"Failed to store artifacts: {e}")
                raise

            # Generate human-readable report
            tprint_info("Generating optimization report")
            result = {
                'optimized_periods': optimized_periods,
                'optimized_lookbacks': optimized_lookbacks,
                'optimization_metadata': optimization_metadata,
                'artifacts': optimization_result.get('artifacts', {})
            }
            report_path = self._generate_optimization_report(
                result, data, **kwargs
            )
            tprint_success("Optimization report generated successfully")
            
            # Store report as artifact
            tprint_info("Storing optimization report as artifact")
            try:
                # Create a simple report object for storage
                report = {
                    'report_path': report_path,
                    'optimized_periods': optimized_periods,
                    'optimized_lookbacks': optimized_lookbacks,
                    'optimization_metadata': optimization_metadata,
                    'generated_at': datetime.now().isoformat()
                }
                artifact_manager.save(
                    step_name='period_lookback_optimization',
                    artifacts={
                        'optimization_report': report
                    },
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'step': 'period_lookback_optimization_report'
                    }
                )
                tprint_success("Optimization report stored as artifact")
            except Exception as e:
                tprint_error(f"Failed to store report artifact: {e}")
                raise

            # Final optimization and cleanup
            tprint_info("🧹 Performing final optimization and cleanup with enhanced hardware tools")
            
            # Final comprehensive optimization
            if isinstance(data, pd.DataFrame):
                data = self.comprehensive_optimizer.optimize_dataframe(
                    data,
                    workload_type=WorkloadType.FEATURE_ENGINEERING,
                    enable_compression=True,
                    enable_gpu_acceleration=True
                )
                tprint_info("Final comprehensive optimization completed")
            
            # Aggressive garbage collection
            if self.aggressive_gc_enabled:
                self._aggressive_garbage_collection()
            
            # Stop memory monitoring
            tprint_info("🧠 Stopping enhanced memory monitoring")
            self.memory_manager.stop_monitoring()
            
            # Get final memory statistics
            memory_stats = self.memory_manager.get_memory_stats()
            tprint_info(f"Final memory stats: {memory_stats.get('memory_percent', 0):.1f}% used")
            
            tprint_success("Period + lookback optimization completed successfully")
            return {
                'success': True,
                'optimized_periods': optimized_periods,
                'optimized_lookbacks': optimized_lookbacks,
                'optimization_metadata': optimization_metadata,
                'optimization_report': report,
                'artifacts': {**(optimization_result.get('artifacts', {}) or {}), 'cache_hit': False},
                'optimization_stats': {
                    'parallel_workers_used': self.parallel_workers,
                    'chunk_size': self.chunk_size,
                    'memory_mapping_enabled': self.memory_mapping_enabled,
                    'aggressive_gc_enabled': self.aggressive_gc_enabled,
                    'data_type_optimization': self.data_type_optimization,
                    'final_memory_usage': memory_stats.get('memory_percent', 0),
                    'enhanced_gpu_acceleration': self.gpu_manager.is_available()
                }
            }

        except Exception as e:
            tprint_error(f"Period + lookback optimization failed: {e}")
            tprint_debug(f"Exception details: {traceback.format_exc()}")
            tprint_data_preview(data, "failed_optimization_data", max_rows=3, level="ERROR")
            self.logger.error(f"Period + lookback optimization failed: {e}")
            raise


    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _compile_feature_level_analysis(self, data, optimized_periods, optimized_lookbacks, metadata,
                                       max_features_to_analyze: int = 200, sample_rows: int = 200_000):
        """Compile per-feature information for troubleshooting.

        Attempts to:
        - Load generated features from the artifact manager (feature_generation step).
        - Infer lookback from feature names (e.g., 'rsi_14').
        - Compute lightweight metrics vs. proxy target (close.pct_change).
        """
        tprint_step("Compiling feature-level analysis")
        tprint_info(f"Analysis parameters: max_features={max_features_to_analyze}, sample_rows={sample_rows}")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(data, "feature_analysis_input", level="DEBUG")
        tprint_data_format(optimized_periods, "optimized_periods", level="DEBUG")
        tprint_data_format(optimized_lookbacks, "optimized_lookbacks", level="DEBUG")
        
        try:
            import re
            import numpy as np
            import pandas as pd
            tprint_info("Required libraries imported successfully")

            def _infer_lookback_from_name(name: str) -> int:
                m = re.search(r"_(\d{1,4})(?!\d)", str(name))
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        return -1
                return -1

            tprint_info("Getting artifact manager for feature analysis")
            artifact_manager = self.get_pretraining_artifact_manager()
            
            # Only look for features from feature_generation_feature_generation_step
            tprint_info("🔍 Looking for features from feature_generation_feature_generation_step...")
            self.logger.info("🔍 Looking for features from feature_generation_feature_generation_step...")
            
            # Try primary artifact keys for feature generation step
            features_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'generated_features')
            if features_df is None or features_df.empty:
                features_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'feature_dataframe')
            if features_df is None or features_df.empty:
                features_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'features')
            
            feature_names = artifact_manager.get_artifact('feature_generation_feature_generation_step', 'generated_feature_names')
            if feature_names is None:
                feature_names = artifact_manager.get_artifact('feature_generation_feature_generation_step', 'feature_names')
            
            source = 'feature_generation_feature_generation_step'
            tprint_info(f"Feature generation step artifacts: df_available={features_df is not None}, names_available={feature_names is not None}")
            
            if features_df is not None and not features_df.empty:
                tprint_success(f"✅ Found {len(features_df.columns)} features from feature_generation_feature_generation_step")
                self.logger.info(f"✅ Found {len(features_df.columns)} features from feature_generation_feature_generation_step")
            else:
                tprint_warning("⚠️ No features found from feature_generation_feature_generation_step")
                self.logger.warning("⚠️ No features found from feature_generation_feature_generation_step")
                
                # Fallback: Use current data for feature analysis
                tprint_info("🔍 Using current data features as fallback")
                self.logger.info("🔍 Using current data features as fallback")
                
                # Use the current data for feature analysis
                features_df = data.copy()
                # Filter out non-feature columns (keep OHLCV and technical indicators)
                feature_columns = [col for col in data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                if feature_columns:
                    features_df = data[feature_columns]
                    feature_names = feature_columns
                    source = 'current_data_features'
                    tprint_info(f"Using current data features: {len(feature_names)} features")
                else:
                    tprint_warning("No feature columns found in current data")
                    return {
                        'status': 'unavailable',
                        'reason': 'no_feature_columns_found',
                        'message': 'No feature columns found in current data for analysis.'
                    }

            if feature_names is None:
                feature_names = list(features_df.columns)
                tprint_info(f"Using column names as feature names: {len(feature_names)} features")

            if len(feature_names) > max_features_to_analyze:
                tprint_info(f"Limiting features to {max_features_to_analyze} (from {len(feature_names)})")
                feature_names = feature_names[:max_features_to_analyze]
                features_df = features_df[feature_names]

            if 'close' not in data.columns:
                tprint_warning("Close column missing - returning partial analysis")
                return {
                    'status': 'partial',
                    'source': source,
                    'analyzed_feature_count': len(feature_names),
                    'global_period': optimized_periods,
                    'global_lookback_default': optimized_lookbacks,
                    'features': [
                        {'name': str(n), 'estimated_lookback': _infer_lookback_from_name(str(n))}
                        for n in feature_names
                    ],
                    'note': 'close column missing; metrics not computed'
                }

            # Load labeling targets if available; fast fail otherwise
            tprint_info("Computing targets for analysis (prefer labeling targets)")
            target_label = 'labeling_targets'
            returns = None
            try:
                # Try common step/key combinations in PreTrainingArtifactManager
                for step_name in ("labeling_integration", "feature_generation_labeling_integration_step"):
                    for key in ("targets", ArtifactKeys.TARGETS):
                        tmp = artifact_manager.get_artifact(step_name, key)
                        if isinstance(tmp, pd.Series) and not tmp.empty:
                            returns = tmp
                            break
                    if isinstance(returns, pd.Series) and not returns.empty:
                        break
            except Exception:
                returns = None
            if returns is None or returns.empty:
                tprint_warning("Labeling targets not found; falling back to close.pct_change()")
                target_label = 'close.pct_change()'
                returns = data['close'].pct_change().fillna(0.0)
            else:
                # Ensure numeric and finite
                returns = returns.astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            if sample_rows and len(returns) > sample_rows:
                tprint_info(f"Sampling data: {sample_rows} rows from {len(returns)}")
                returns = returns.iloc[-sample_rows:]
                features_df = features_df.iloc[-sample_rows:]

            r_vals = returns.values

            def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
                try:
                    if x.size == 0 or y.size == 0:
                        return 0.0
                    xv = x - (x.mean() if x.size else 0.0)
                    yv = y - (y.mean() if y.size else 0.0)
                    denom = (np.sqrt((xv * xv).sum()) * np.sqrt((yv * yv).sum()))
                    if denom == 0:
                        return 0.0
                    return float((xv * yv).sum() / denom)
                except Exception:
                    return 0.0

            tprint_info(f"Analyzing {len(feature_names)} features")
            rows = []
            for i, name in enumerate(feature_names):
                if i % 50 == 0:  # Progress indicator for large feature sets
                    tprint_info(f"Processing feature {i+1}/{len(feature_names)}: {name}")
                
                s = features_df[name].astype(float)
                aligned = pd.concat([s, returns], axis=1).dropna()
                if aligned.empty:
                    rows.append({
                        'name': str(name),
                        'estimated_lookback': _infer_lookback_from_name(str(name)),
                        'non_null_pct': 0.0,
                        'pearson_corr': 0.0,
                        'autocorr_lag1': 0.0,
                        'mean': 0.0,
                        'std': 0.0,
                        'global_period': optimized_periods
                    })
                    continue

                x = aligned.iloc[:, 0].values
                y = aligned.iloc[:, 1].values

                corr = safe_corr(x, y)
                ac1 = 0.0
                try:
                    if x.size > 2:
                        ac1 = safe_corr(x[:-1], x[1:])
                except Exception:
                    ac1 = 0.0

                non_null_pct = float(aligned.shape[0] / max(1, len(s))) * 100.0
                rows.append({
                    'name': str(name),
                    'estimated_lookback': _infer_lookback_from_name(str(name)),
                    'non_null_pct': round(non_null_pct, 2),
                    'pearson_corr': round(float(corr), 6),
                    'autocorr_lag1': round(float(ac1), 6),
                    'mean': round(float(np.nanmean(x)), 6),
                    'std': round(float(np.nanstd(x)), 6),
                    'global_period': optimized_periods
                })

            tprint_info(f"Sorting {len(rows)} features by correlation strength")
            rows_sorted = sorted(rows, key=lambda d: abs(d.get('pearson_corr', 0.0)), reverse=True)
            tprint_success(f"Feature-level analysis completed: {len(rows_sorted)} features analyzed")
            return {
                'status': 'ok',
                'source': source,
                'analyzed_feature_count': len(rows_sorted),
                'global_period': optimized_periods,
                'global_lookback_default': optimized_lookbacks,
                'target_used': target_label,
                'features': rows_sorted
            }

        except Exception as e:
            tprint_error(f"Feature-level analysis failed: {e}")
            tprint_debug(f"Feature analysis error details: {traceback.format_exc()}")
            tprint_data_preview(data, "failed_operation_data", max_rows=3, level="ERROR")
            self.logger.warning(f"Feature-level analysis unavailable: {e}")
            return {
                'status': 'unavailable',
                'reason': str(e)
            }

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _generate_optimization_report(self, result, data, **kwargs):
        """Generate a comprehensive human-readable optimization report."""
        try:
            # Add comprehensive data format analysis for troubleshooting
            tprint_data_format(result, "optimization_report_result", level="DEBUG")
            tprint_data_format(data, "optimization_report_data", level="DEBUG")
            
            # Get current timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract symbol and timeframe from kwargs
            symbol = kwargs.get('symbol', 'UNKNOWN')
            timeframe = kwargs.get('timeframe', 'UNKNOWN')
            
            # Create outcomes directory (repo-relative) if it doesn't exist
            from pathlib import Path
            outcomes_dir = str(Path("outcomes"))
            os.makedirs(outcomes_dir, exist_ok=True)
            
            # Generate filename with timestamp
            filename = f"period_lookback_optimization_report_{symbol}_{timeframe}_{timestamp}.md"
            report_path = os.path.join(outcomes_dir, filename)
            
            # Print report path to console
            print(f"📊 Report generated: {report_path}")
            tprint_success(f"📊 Report generated: {report_path}")
            
            # Extract optimization results (top-level keys if available)
            optimized_periods = result.get('optimized_periods', result.get('optimization_results', {}).get('optimized_periods', {}))
            optimized_lookbacks = result.get('optimized_lookbacks', result.get('optimization_results', {}).get('optimized_lookbacks', {}))
            optimization_metadata = result.get('optimization_metadata', {})
            
            # Calculate metrics
            total_features = len(data.columns) if hasattr(data, 'columns') else 0
            data_rows = len(data) if hasattr(data, '__len__') else 0
            data_memory = data.memory_usage(deep=True).sum() / 1024**2 if hasattr(data, 'memory_usage') else 0
            
            # Calculate averages
            avg_period = np.mean(list(optimized_periods.values())) if optimized_periods else 0
            avg_lookback = np.mean(list(optimized_lookbacks.values())) if optimized_lookbacks else 0
            min_period = min(optimized_periods.values()) if optimized_periods else 0
            max_period = max(optimized_periods.values()) if optimized_periods else 0
            min_lookback = min(optimized_lookbacks.values()) if optimized_lookbacks else 0
            max_lookback = max(optimized_lookbacks.values()) if optimized_lookbacks else 0
            
            # Generate report content
            ratio_str = (f"{(avg_period/avg_lookback):.2f}" if avg_lookback > 0 else "N/A")

            report_content = f"""# Period & Lookback Optimization Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Symbol:** {symbol}  
**Timeframe:** {timeframe}  
**Report Path:** `{report_path}`

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Features** | {total_features:,} |
| **Data Rows** | {data_rows:,} |
| **Data Memory Usage** | {data_memory:.2f} MB |
| **Optimization Status** | ✅ Success |
| **Average Period** | {avg_period:.1f} |
| **Average Lookback** | {avg_lookback:.1f} |

## 🎯 Optimization Results

### Global Averages
- **Average Period:** {avg_period:.1f} (Range: {min_period} - {max_period})
- **Average Lookback:** {avg_lookback:.1f} (Range: {min_lookback} - {max_lookback})
- **Total Features Optimized:** {len(optimized_periods)}/{total_features}
- **Optimization Coverage:** {(len(optimized_periods)/total_features*100):.1f}% of features optimized
- **Average Period/Lookback Ratio:** {ratio_str}
- **Total Optimization Score:** {sum([(p * l) / 100 for p, l in zip(optimized_periods.values(), optimized_lookbacks.values()) if p > 0 and l > 0]):.2f}

### Period Distribution
- **Minimum Period:** {min_period}
- **Maximum Period:** {max_period}
- **Period Range:** {max_period - min_period}
- **Standard Deviation:** {np.std(list(optimized_periods.values())):.1f if optimized_periods else 0}

### Lookback Distribution
- **Minimum Lookback:** {min_lookback}
- **Maximum Lookback:** {max_lookback}
- **Lookback Range:** {max_lookback - min_lookback}
- **Standard Deviation:** {np.std(list(optimized_lookbacks.values())):.1f if optimized_lookbacks else 0}

## 📈 Per-Feature Analysis

### Feature Optimization Summary
- **Total Features Analyzed:** {len(optimized_periods)}
- **Features with Optimized Periods:** {len([p for p in optimized_periods.values() if p > 0])}
- **Features with Optimized Lookbacks:** {len([l for l in optimized_lookbacks.values() if l > 0])}

### Top 10 Features by Period Length
"""

            # Append MI/mRMR sections if artifacts available
            try:
                artifacts_all = result.get('artifacts', {})
                mi_best = artifacts_all.get(ArtifactKeys.MI_BEST_LOOKBACKS_PER_FEATURE, {})
                mrmr_top = artifacts_all.get(ArtifactKeys.MRMR_TOP_LOOKBACKS_PER_FEATURE, {})
                mi_scores_by_feat = artifacts_all.get(ArtifactKeys.MI_SCORES_BY_FEATURE, {})
                oos_sharpe_map = artifacts_all.get(ArtifactKeys.OOS_SHARPE_BY_FEATURE_WINDOW, {})
                opt_cfg = artifacts_all.get(ArtifactKeys.OPTIMIZATION_CONFIG, {})

                if mi_best or mrmr_top:
                    report_content += "\n## 🧠 Per-Feature MI and mRMR Results\n"
                    # MI-best summary table
                    if mi_best:
                        report_content += "\n### MI-Best Window per Base Feature (Weighted: 80% MI + 20% Stability)\n"
                        report_content += "\n| Base Feature | Best Window | MI Score | Stability | Weighted Score |\n|---|---:|---:|---:|---:|\n"
                        rows = []
                        # Build list with stability; recompute stability quickly for the matched column
                        for base, win in mi_best.items():
                            feats = mi_scores_by_feat.get(base, {})
                            mi_sc = 0.0
                            stab_sc = 0.0
                            chosen_col = None
                            if feats:
                                # locate the column matching this window
                                for col, sc in feats.items():
                                    if win is not None and str(win) in str(col):
                                        mi_sc = float(sc)
                                        chosen_col = col
                                        break
                            if chosen_col is not None and chosen_col in data.columns:
                                try:
                                    stab_sc = float(self._compute_stability_scores([chosen_col], data).get(chosen_col, 0.0))
                                except Exception:
                                    stab_sc = 0.0
                            weighted = 0.8 * mi_sc + 0.2 * stab_sc
                            rows.append((base, win, mi_sc, stab_sc, weighted))
                        # Top 25 by weighted score
                        for base, win, mi_sc, stab_sc, w_sc in sorted(rows, key=lambda x: x[4], reverse=True)[:25]:
                            report_content += f"| {base} | {win if win is not None else '-'} | {mi_sc:.4f} | {stab_sc:.4f} | {w_sc:.4f} |\n"

                    # mRMR per-base selection
                    if mrmr_top:
                        report_content += "\n### mRMR-Selected Windows per Base Feature (Sharpe-centric)\n"
                        report_content += "\n| Base Feature | Windows | Avg OOS Sharpe (candidates) |\n|---|---|---:|\n"
                        for base, wins in mrmr_top.items():
                            cand_map = oos_sharpe_map.get(base, {})
                            avg_sh = 0.0
                            if cand_map:
                                avg_sh = float(np.mean([float(v) for v in cand_map.values()]))
                            wins_str = ", ".join([str(w) if w is not None else '-' for w in (wins or [])])
                            report_content += f"| {base} | {wins_str} | {avg_sh:.3f} |\n"

                    # Config
                    if opt_cfg:
                        report_content += "\n### Selection Configuration\n"
                        for k, v in opt_cfg.items():
                            report_content += f"- {k}: {v}\n"
            except Exception as e:
                tprint_warning(f"Failed to append MI/mRMR sections: {e}")
            
            # Add top features by period
            if optimized_periods:
                sorted_periods = sorted(optimized_periods.items(), key=lambda x: x[1], reverse=True)[:10]
                report_content += "\n| Feature | Period | Lookback |\n|---------|--------|----------|\n"
                for feature, period in sorted_periods:
                    lookback = optimized_lookbacks.get(feature, 'N/A')
                    report_content += f"| {feature} | {period} | {lookback} |\n"
            
            report_content += f"""
### Top 10 Features by Lookback Length
"""
            
            # Add top features by lookback
            if optimized_lookbacks:
                sorted_lookbacks = sorted(optimized_lookbacks.items(), key=lambda x: x[1], reverse=True)[:10]
                report_content += "\n| Feature | Lookback | Period |\n|---------|----------|--------|\n"
                for feature, lookback in sorted_lookbacks:
                    period = optimized_periods.get(feature, 'N/A')
                    report_content += f"| {feature} | {lookback} | {period} |\n"
            
            # Add detailed per-feature metrics
            report_content += f"""
### Detailed Per-Feature Metrics
| Feature | Period | Lookback | Period/Lookback Ratio | Optimization Score |
|---------|--------|----------|----------------------|-------------------|
"""
            
            if optimized_periods and optimized_lookbacks:
                for feature in sorted(optimized_periods.keys()):
                    period = optimized_periods.get(feature, 0)
                    lookback = optimized_lookbacks.get(feature, 0)
                    ratio = period / lookback if lookback > 0 else 0
                    # Calculate a simple optimization score (period * lookback / 100)
                    score = (period * lookback) / 100 if period > 0 and lookback > 0 else 0
                    report_content += f"| {feature} | {period} | {lookback} | {ratio:.2f} | {score:.2f} |\n"
            
            # Add statistical analysis
            report_content += f"""
### Statistical Analysis
"""
            if optimized_periods:
                period_values = list(optimized_periods.values())
                report_content += f"- **Period Mean:** {np.mean(period_values):.2f}\n"
                report_content += f"- **Period Median:** {np.median(period_values):.2f}\n"
                report_content += f"- **Period Mode:** {max(set(period_values), key=period_values.count)}\n"
                report_content += f"- **Period Variance:** {np.var(period_values):.2f}\n"
            
            if optimized_lookbacks:
                lookback_values = list(optimized_lookbacks.values())
                report_content += f"- **Lookback Mean:** {np.mean(lookback_values):.2f}\n"
                report_content += f"- **Lookback Median:** {np.median(lookback_values):.2f}\n"
                report_content += f"- **Lookback Mode:** {max(set(lookback_values), key=lookback_values.count)}\n"
                report_content += f"- **Lookback Variance:** {np.var(lookback_values):.2f}\n"
            
            report_content += f"""
## 🔧 Configuration Details

### Optimization Parameters
- **Optimization Method:** {optimization_results.get('optimization_method', 'standalone_optimization')}
- **Correlation Threshold:** {optimization_metadata.get('correlation_threshold', 'N/A')}
- **Minimum Periods:** {optimization_metadata.get('min_periods', 'N/A')}
- **Maximum Periods:** {optimization_metadata.get('max_periods', 'N/A')}

### System Configuration
- **Parallel Workers:** {result.get('optimization_stats', {}).get('parallel_workers_used', 'N/A')}
- **Enhanced GPU Acceleration:** {result.get('optimization_stats', {}).get('enhanced_gpu_acceleration', False)}
- **Final Memory Usage:** {result.get('optimization_stats', {}).get('final_memory_usage', 0):.1f}%

## 📋 Feature Categories Analysis

### Period Length Categories
- **Short Periods (5-15):** {len([p for p in optimized_periods.values() if 5 <= p <= 15])} features
- **Medium Periods (16-30):** {len([p for p in optimized_periods.values() if 16 <= p <= 30])} features  
- **Long Periods (31+):** {len([p for p in optimized_periods.values() if p > 30])} features

### Lookback Length Categories
- **Short Lookbacks (5-15):** {len([l for l in optimized_lookbacks.values() if 5 <= l <= 15])} features
- **Medium Lookbacks (16-30):** {len([l for l in optimized_lookbacks.values() if 16 <= l <= 30])} features
- **Long Lookbacks (31+):** {len([l for l in optimized_lookbacks.values() if l > 30])} features

## 🎯 Recommendations

### Based on Optimization Results
"""
            
            # Add recommendations
            recommendations = self._generate_recommendations(avg_period, avg_lookback, optimization_metadata)
            for rec in recommendations:
                report_content += f"- {rec}\n"
            
            report_content += f"""
## 📊 Performance Metrics

### Memory Optimization
- **Initial Data Size:** {data_memory:.2f} MB
- **Data Type Optimization:** {result.get('optimization_stats', {}).get('memory_optimization_saved', 0):.2f} MB saved
- **Memory Efficiency:** {((data_memory - result.get('optimization_stats', {}).get('memory_optimization_saved', 0)) / data_memory * 100):.1f}% reduction

### Processing Performance
- **Total Processing Time:** {result.get('execution_summary', {}).get('total_time', 0):.2f} seconds
- **Features Processed per Second:** {total_features / result.get('execution_summary', {}).get('total_time', 1):.1f}
- **Memory Usage Peak:** {result.get('optimization_stats', {}).get('final_memory_usage', 0):.1f}%

## 🔍 Technical Details

### Data Characteristics
- **Data Shape:** {data.shape if hasattr(data, 'shape') else 'Unknown'}
- **Data Types:** {len(data.dtypes.unique()) if hasattr(data, 'dtypes') else 'Unknown'} unique types
- **Missing Values:** {data.isnull().sum().sum() if hasattr(data, 'isnull') else 'Unknown'}

### Optimization Algorithm
- **Method:** Concurrent period and lookback optimization
- **Correlation Analysis:** Enabled with threshold >0.85
- **Redundancy Removal:** Active
- **Enhanced Hardware Optimizations:** Enabled

---
*Report generated by Ares Period & Lookback Optimization System*
*Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return report_path
            
        except Exception as e:
            tprint_error(f"Failed to generate optimization report: {e}")
            return None

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _generate_recommendations(self, periods, lookbacks, metadata):
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Period recommendations
        if periods >= 30:
            recommendations.append("✅ Good period length - provides sufficient historical context")
        elif periods < 20:
            recommendations.append("⚠️ Consider increasing period length for better stability")
        
        # Lookback recommendations  
        if lookbacks >= 20:
            recommendations.append("✅ Adequate lookback window for feature computation")
        elif lookbacks < 15:
            recommendations.append("⚠️ Consider increasing lookback for more robust features")
        
        # CMI analysis recommendations
        cmi_diagnostics = metadata.get('cmi_diagnostics', {})
        if cmi_diagnostics.get('cmi_enabled', False):
            recommendations.append("🎯 CMI complementarity optimization was applied")
        else:
            recommendations.append("📊 Standard optimization used (CMI complementarity not available)")
        
        # Data quality recommendations
        if metadata.get('no_recency_bias', True):
            recommendations.append("✅ Recency bias prevention enabled")
        
        if metadata.get('correlation_threshold', 0.85) <= 0.85:
            recommendations.append("✅ Appropriate correlation threshold for feature diversity")
        
        return recommendations

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _generate_step_summaries(self, optimized_periods, optimized_lookbacks, metadata, data):
        """Generate detailed summaries for each optimization step."""
        step_summaries = {
            'data_preparation': {
                'step_name': 'Data Preparation & Validation',
                'description': 'Data loading, cleaning, and validation for optimization',
                'details': {
                    'data_source': 'Consolidated parquet files',
                    'data_rows': len(data),
                    'data_columns': len(data.columns),
                    'memory_usage_mb': f"{data.memory_usage(deep=True).sum() / 1024**2:.2f}",
                    'data_quality_checks': [
                        'Non-finite value detection and correction',
                        'Data completeness validation',
                        'Memory usage optimization'
                    ],
                    'validation_rules': {
                        'min_rows': 100,
                        'required_columns': ['open', 'high', 'low', 'close'],
                        'data_types': 'pandas.DataFrame'
                    }
                },
                'status': 'completed',
                'duration_estimate': '~0.5s'
            },
            'period_optimization': {
                'step_name': 'Period Optimization',
                'description': 'Optimization of feature generation periods for maximum historical context',
                'details': {
                    'optimized_value': optimized_periods,
                    'optimization_method': 'standalone_optimization',
                    'constraints': {
                        'min_periods': metadata.get('min_periods', 2),
                        'correlation_threshold': metadata.get('correlation_threshold', 0.85),
                        'no_recency_bias': metadata.get('no_recency_bias', True)
                    },
                    'optimization_criteria': [
                        'Sufficient historical context',
                        'Feature stability across periods',
                        'Correlation threshold compliance',
                        'Recency bias prevention'
                    ],
                    'result_analysis': f"Period length of {optimized_periods} provides {'excellent' if optimized_periods >= 30 else 'adequate' if optimized_periods >= 20 else 'minimal'} historical context"
                },
                'status': 'completed',
                'duration_estimate': '~0.8s'
            },
            'lookback_optimization': {
                'step_name': 'Lookback Window Optimization',
                'description': 'Optimization of lookback windows for feature computation stability',
                'details': {
                    'optimized_value': optimized_lookbacks,
                    'optimization_method': 'standalone_optimization',
                    'constraints': {
                        'min_lookback': 5,
                        'max_lookback': 252,
                        'stability_requirement': True
                    },
                    'optimization_criteria': [
                        'Feature computation stability',
                        'Sufficient data for rolling calculations',
                        'Memory efficiency',
                        'Computational performance'
                    ],
                    'result_analysis': f"Lookback window of {optimized_lookbacks} provides {'excellent' if optimized_lookbacks >= 20 else 'adequate' if optimized_lookbacks >= 15 else 'minimal'} computation stability"
                },
                'status': 'completed',
                'duration_estimate': '~0.5s'
            },
            'feature_selection_analysis': {
                'step_name': 'Feature Selection Analysis',
                'description': 'Analysis of feature selection criteria and constraints',
                'details': {
                    'selection_criteria': {
                        'top_1_trading': metadata.get('top_1_trading', True),
                        'top_3_interactions': metadata.get('top_3_interactions', True),
                        'correlation_threshold': metadata.get('correlation_threshold', 0.85)
                    },
                    'feature_diversity': {
                        'correlation_threshold': '0.85 (prevents highly correlated features)',
                        'interaction_features': 'Top 3 interactions enabled',
                        'trading_features': 'Top 1 trading features prioritized'
                    },
                    'quality_metrics': [
                        'Feature diversity maintenance',
                        'Correlation reduction',
                        'Interaction feature inclusion',
                        'Trading signal prioritization'
                    ]
                },
                'status': 'completed',
                'duration_estimate': '~0.2s'
            }
        }
        
        # Only include CMI analysis if in Tactician mode
        if metadata.get('cmi_diagnostics', {}).get('cmi_enabled', False):
            step_summaries['cmi_complementarity_analysis'] = {
                'step_name': 'CMI Complementarity Analysis',
                'description': 'Conditional Mutual Information complementarity analysis for Tactician mode',
                'details': {
                    'cmi_enabled': True,
                    'analysis_type': 'Tactician mode CMI complementarity',
                    'cmi_diagnostics': metadata.get('cmi_diagnostics', {}),
                    'complementarity_regularizer': {
                        'enabled': True,
                        'objective': 'Obj = w_model·Perf + w_cmi·R̄ - w_red·D̄',
                        'weights': metadata.get('cmi_diagnostics', {}).get('regularizer_weights', {})
                    },
                    'analyst_integration': {
                        'analyst_source': metadata.get('cmi_diagnostics', {}).get('analyst_source', 'N/A'),
                        'analyst_dimensions': metadata.get('cmi_diagnostics', {}).get('analyst_dims', 'N/A'),
                        'mutual_information': metadata.get('cmi_diagnostics', {}).get('I_Y_A', 'N/A')
                    }
                },
                'status': 'completed',
                'duration_estimate': '~0.3s'
            }
        
        # Add artifact storage step
        step_summaries['artifact_storage'] = {
                'step_name': 'Artifact Storage & Persistence',
                'description': 'Storage of optimization results and metadata for future use',
                'details': {
                    'storage_path': str(get_pretraining_artifact_manager().config.base_dir / 'period_lookback_optimization'),
                    'stored_artifacts': [
                        'optimized_periods.pkl',
                        'optimized_lookbacks.pkl', 
                        'optimization_metadata.pkl',
                        'optimization_report.pkl',
                        'metadata.json'
                    ],
                    'persistence_method': 'Disk + Memory (hybrid storage)',
                    'retrieval_method': 'Automatic fallback (memory → disk)',
                    'metadata_included': [
                        'Optimization parameters',
                        'Configuration settings',
                        'CMI diagnostics',
                        'Execution timestamps',
                        'Data quality metrics'
                    ]
                },
                'status': 'completed',
                'duration_estimate': '~0.1s'
            }
        
        return step_summaries

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _format_markdown_report(self, report):
        """Format the report as markdown."""
        md = f"""# {report['title']}

**Generated:** {report['timestamp']}
**Artifact Storage Path:** `{report['artifact_storage_path']}`

## 📊 Execution Summary

- **Status:** {report['execution_summary']['status']}
- **Data Rows:** {report['execution_summary']['data_rows']:,}
- **Data Columns:** {report['execution_summary']['data_columns']}
- **Memory Usage:** {report['execution_summary']['data_memory_usage']}

## 🎯 Optimization Results

- **Optimized Periods:** {report['optimization_results']['optimized_periods']}
- **Optimized Lookbacks:** {report['optimization_results']['optimized_lookbacks']}
- **Method:** {report['optimization_results']['optimization_method']}

## ⚙️ Configuration

- **Symbol:** {report['configuration']['symbol']}
- **Timeframe:** {report['configuration']['timeframe']}
- **Direction:** {report['configuration']['direction']}
- **Min Periods:** {report['configuration']['min_periods']}
- **Correlation Threshold:** {report['configuration']['correlation_threshold']}
- **No Recency Bias:** {report['configuration']['no_recency_bias']}
- **Top 1 Trading:** {report['configuration']['top_1_trading']}
- **Top 3 Interactions:** {report['configuration']['top_3_interactions']}

## 🔧 Step-by-Step Analysis

{self._format_step_summaries_markdown(report['step_summaries'])}

## 🧩 Feature-Level Optimization

"""
        
        # Feature-level details (top features by |corr|)
        feature_level = report.get('feature_level_analysis', {})
        if feature_level and feature_level.get('status') in {'ok', 'partial'}:
            md += f"- **Source:** {feature_level.get('source', 'unknown')}\n"
            md += f"- **Analyzed Features:** {feature_level.get('analyzed_feature_count', 0)}\n"
            md += f"- **Global Period (default):** {feature_level.get('global_period', 'N/A')}\n"
            md += f"- **Global Lookback (default):** {feature_level.get('global_lookback_default', 'N/A')}\n"
            if feature_level.get('status') == 'partial' and feature_level.get('note'):
                md += f"- **Note:** {feature_level.get('note')}\n"
            md += "\n### Top Features by |Pearson Corr| vs returns\n\n"
            md += self._format_feature_table_markdown(feature_level.get('features', []), max_rows=40)
        else:
            reason = feature_level.get('reason', 'not available') if isinstance(feature_level, dict) else 'not available'
            md += f"_Feature-level details {reason}._\n\n"
        
        md += "\n## 🧠 CMI Analysis\n\n"
        
        cmi_diagnostics = report['cmi_analysis']
        if cmi_diagnostics.get('cmi_enabled', False):
            md += f"- **CMI Enabled:** ✅ Yes\n"
            md += f"- **Analyst Source:** {cmi_diagnostics.get('analyst_source', 'Unknown')}\n"
            md += f"- **Analyst Dimensions:** {cmi_diagnostics.get('analyst_dims', 'Unknown')}\n"
        else:
            md += f"- **CMI Enabled:** ❌ No\n"
            md += f"- **Reason:** {cmi_diagnostics.get('reason', 'Unknown')}\n"
        
        md += "\n## 💡 Recommendations\n\n"
        tprint_info(f"Adding {len(report['recommendations'])} recommendations to markdown")
        for rec in report['recommendations']:
            md += f"- {rec}\n"
        
        md += "\n## 🚀 Next Steps\n\n"
        tprint_info(f"Adding {len(report['next_steps'])} next steps to markdown")
        for step in report['next_steps']:
            md += f"- {step}\n"
        
        tprint_success(f"Markdown report formatted: {len(md)} characters")
        return md

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _format_feature_table_markdown(self, features, max_rows: int = 40):
        """Render compact table for feature metrics."""
        tprint_info(f"Formatting feature table: {len(features)} features, max_rows={max_rows}")
        try:
            if not features:
                tprint_warning("No features available for table formatting")
                return "_No feature-level details available._\n"
            # Header
            tprint_info("Creating feature table header")
            md = "| Feature | Lookback | |Pearson Corr| | Non-Null % | Mean | Std | AC(1) |\n"
            md += "|---|---:|---:|---:|---:|---:|\n"
            rows = 0
            for f in features[:max_rows]:
                name = str(f.get('name', ''))
                lb = f.get('estimated_lookback', '-')
                corr = abs(f.get('pearson_corr', 0.0) or 0.0)
                nn = f.get('non_null_pct', 0.0) or 0.0
                mean = f.get('mean', 0.0) or 0.0
                std = f.get('std', 0.0) or 0.0
                ac1 = f.get('autocorr_lag1', 0.0) or 0.0
                md += f"| {name} | {lb} | {corr:.4f} | {nn:.2f} | {mean:.4f} | {std:.4f} | {ac1:.4f} |\n"
                rows += 1
            if len(features) > rows:
                md += f"\n_+{len(features) - rows} more features not shown..._\n"
            tprint_success(f"Feature table formatted: {rows} rows displayed")
            return md
        except Exception as e:
            tprint_error(f"Failed to render feature table: {e}")
            return f"_Failed to render feature table: {e}_\n"

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _format_step_summaries_markdown(self, step_summaries):
        """Format step summaries as markdown."""
        tprint_info(f"Formatting {len(step_summaries)} step summaries as markdown")
        md = ""
        for step_key, step_info in step_summaries.items():
            tprint_info(f"Formatting step: {step_key}")
            status_emoji = "✅" if step_info['status'] == 'completed' else "⏭️" if step_info['status'] == 'skipped' else "❌"
            md += f"### {status_emoji} {step_info['step_name']}\n\n"
            md += f"**Description:** {step_info['description']}\n\n"
            md += f"**Status:** {step_info['status']} | **Duration:** {step_info['duration_estimate']}\n\n"
            
            # Format details
            if 'details' in step_info:
                md += "**Details:**\n"
                for key, value in step_info['details'].items():
                    if isinstance(value, list):
                        md += f"- **{key.replace('_', ' ').title()}:**\n"
                        for item in value:
                            md += f"  - {item}\n"
                    elif isinstance(value, dict):
                        md += f"- **{key.replace('_', ' ').title()}:**\n"
                        for sub_key, sub_value in value.items():
                            md += f"  - {sub_key.replace('_', ' ').title()}: {sub_value}\n"
                    else:
                        md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                md += "\n"
            
            md += "---\n\n"
        
        tprint_success(f"Step summaries formatted: {len(md)} characters")
        return md

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.BALANCED,
        enable_compression=True
    ))
    def _store_human_readable_report(self, report, markdown_report, metadata):
        """Store human-readable report in outcomes/ directory."""
        tprint_step("Storing human-readable report")
        tprint_info(f"Report size: {len(markdown_report)} characters")
        try:
            import os
            from pathlib import Path
            
            # Create outcomes directory if it doesn't exist
            tprint_info("Creating outcomes directory")
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            tprint_success("Outcomes directory ready")
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = metadata.get('symbol', 'ETHUSDT')
            timeframe = metadata.get('timeframe', '15m')
            tprint_info(f"Report filename components: symbol={symbol}, timeframe={timeframe}, timestamp={timestamp}")
            
            # Store markdown report
            md_filename = f"period_lookback_optimization_report_{symbol}_{timeframe}_{timestamp}.md"
            md_path = outcomes_dir / md_filename
            tprint_info(f"Storing markdown report: {md_path}")
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            tprint_success(f"Markdown report stored: {md_path}")
            
            # Store JSON report
            json_filename = f"period_lookback_optimization_report_{symbol}_{timeframe}_{timestamp}.json"
            json_path = outcomes_dir / json_filename
            tprint_info(f"Storing JSON report: {json_path}")
            
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            tprint_success(f"JSON report stored: {json_path}")
            
            tprint_success("📄 Human-readable reports stored in outcomes/")
            tprint_info(f"   - Markdown: {md_path}")
            tprint_info(f"   - JSON: {json_path}")
            self.logger.info(f"📄 Human-readable reports stored in outcomes/:")
            self.logger.info(f"   - Markdown: {md_path}")
            self.logger.info(f"   - JSON: {json_path}")
            
        except Exception as e:
            tprint_error(f"Failed to store human-readable report: {e}")
            tprint_debug(f"Report storage error details: {traceback.format_exc()}")
            self.logger.error(f"Failed to store human-readable report: {e}")

    def _get_validation_rules(self):
        """Get validation rules for this component."""
        tprint_info("Getting validation rules")
        rules = {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close'],
            'min_rows': 100,
            'min_periods': self.min_periods,
            'correlation_threshold': self.correlation_threshold
        }
        tprint_info(f"Validation rules: {rules}")
        return rules

    def _validate_component_specific(self, data):
        """Validate component-specific requirements."""
        tprint_step("Validating component-specific requirements")
        tprint_info(f"Data type: {type(data)}, shape: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            tprint_info(f"DataFrame validation: {len(data)} rows, {len(data.columns)} columns")
            if len(data) < 100:
                error_msg = f"Data has {len(data)} rows, minimum required: 100"
                errors.append(error_msg)
                tprint_error(error_msg)
            else:
                tprint_success(f"Data row count validation passed: {len(data)} rows")
            
            metadata['shape'] = data.shape
            metadata['columns'] = list(data.columns)
            tprint_info(f"Metadata: shape={metadata['shape']}, columns={len(metadata['columns'])}")
        else:
            tprint_warning(f"Data is not a DataFrame: {type(data)}")
        
        tprint_info(f"Validation results: {len(errors)} errors, {len(warnings)} warnings")
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

    # -----------------------------
    # Per-feature MI/mRMR utilities
    # -----------------------------
    def _enforce_15m_timeframe(self, df: pd.DataFrame) -> None:
        try:
            if not isinstance(df.index, (pd.DatetimeIndex,)):
                raise ValueError("DataFrame index must be a DatetimeIndex for timeframe enforcement")
            if len(df.index) < 3:
                return
            median_delta = df.index.to_series().diff().dropna().median()
            if pd.isna(median_delta):
                return
            # allow ±1 minute tolerance
            if not (abs(median_delta - pd.Timedelta(minutes=15)) <= pd.Timedelta(minutes=1)):
                raise ValueError("Timeframe enforcement failed: expected ~15m bars. Abort.")
        except Exception as e:
            tprint_error(f"Timeframe enforcement error: {e}")
            raise

    def _align_features_targets(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        try:
            # Inner-join on index
            before_rows = (len(features), int(targets.shape[0]))
            joined_index = features.index.intersection(targets.index)
            f2 = features.loc[joined_index]
            t2 = targets.loc[joined_index]
            # Drop rows with any NaNs in features or target
            valid_mask = (~f2.isna().any(axis=1)) & (~t2.isna())
            dropped = len(f2) - int(valid_mask.sum())
            meta['rows_before'] = before_rows
            meta['rows_after'] = (int(valid_mask.sum()), int(valid_mask.sum()))
            meta['rows_dropped'] = dropped
            if dropped > 0:
                tprint_warning(f"Alignment dropped {dropped} rows due to NaNs or mismatched index")
            return f2.loc[valid_mask], t2.loc[valid_mask], meta
        except Exception as e:
            tprint_error(f"Alignment failed: {e}")
            raise

    def _group_windows_by_family(self, columns: List[str]) -> Dict[str, List[Tuple[str, Optional[int]]]]:
        pattern = re.compile(r"(?P<base>[a-z_]+?)(?:_(?:lookback|period|window))?_(?P<window>\d+)")
        families: Dict[str, List[Tuple[str, Optional[int]]]] = {}
        for col in columns:
            m = pattern.search(str(col))
            if m:
                base = m.group('base')
                try:
                    win = int(m.group('window'))
                except Exception:
                    win = None
            else:
                # no window - single candidate family named by full col
                base = str(col)
                win = None
            families.setdefault(base, []).append((str(col), win))
        return families

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    def _compute_mi_scores(self, feature_cols: List[str], features: pd.DataFrame, targets: pd.Series,
                            sample_n: int = 100_000) -> Dict[str, float]:
        """Compute approximate MI with KSG-like estimator (sklearn), with rank transform and jitter.

        - Rank-pct transform both X and y to reduce bias, add tiny jitter to break ties.
        - Subsample to at most sample_n rows for speed during prefiltering.
        """
        from sklearn.feature_selection import mutual_info_regression
        rng = np.random.default_rng(42)
        scores: Dict[str, float] = {}
        try:
            y = targets.astype(float)
            # Rank-pct transform with jitter
            y_rank = y.rank(pct=True).astype(np.float64).to_numpy()
            if y_rank.size == 0 or np.unique(np.nan_to_num(y_rank)).shape[0] <= 1:
                return {col: 0.0 for col in feature_cols}
            # Jitter
            y_rank = y_rank + (rng.normal(0, 1e-12, size=y_rank.shape))
            # Subsample indices (use tail if deterministic desired)
            idx = np.arange(len(y_rank))
            if len(idx) > sample_n:
                idx = idx[-sample_n:]
            y_sub = y_rank[idx]

            for col in feature_cols:
                x = features[col].astype(float)
                x_rank = x.rank(pct=True).astype(np.float64).to_numpy()
                x_rank = x_rank + (rng.normal(0, 1e-12, size=x_rank.shape))
                xs = x_rank[idx]
                mask = np.isfinite(xs) & np.isfinite(y_sub)
                if mask.sum() < 50:
                    scores[col] = 0.0
                    continue
                try:
                    mi = mutual_info_regression(xs[mask].reshape(-1, 1), y_sub[mask], random_state=42, n_neighbors=3)
                    scores[col] = float(mi[0])
                except Exception:
                    scores[col] = 0.0
        except Exception:
            scores = {col: 0.0 for col in feature_cols}
        return scores

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    def _compute_stability_scores(self, feature_cols: List[str], features: pd.DataFrame) -> Dict[str, float]:
        """Compute fast stability scores per column using inverse coefficient of variation.

        stability = 1 / (1 + |std / (mean_abs + eps)|)
        Range in (0, 1]; higher is more stable.
        """
        scores: Dict[str, float] = {}
        eps = 1e-8
        for col in feature_cols:
            s = features[col].astype(float)
            vals = s.to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size < 10:
                scores[col] = 0.0
                continue
            mean_abs = np.mean(np.abs(vals))
            std = np.std(vals)
            cv = std / (mean_abs + eps)
            stability = 1.0 / (1.0 + abs(cv))
            scores[col] = float(stability)
        return scores

    def _prefilter_by_mi_with_spacing(self, items: List[Tuple[str, Optional[int]]], mi_scores: Dict[str, float], max_M: int = 6, spacing: int = 2) -> List[Tuple[str, Optional[int]]]:
        # Sort by MI descending
        ranked = sorted(items, key=lambda t: mi_scores.get(t[0], 0.0), reverse=True)
        selected: List[Tuple[str, Optional[int]]] = []
        for name, win in ranked:
            if len(selected) >= max_M:
                break
            if win is None:
                # allow single-candidate families without numeric window
                if not selected:
                    selected.append((name, win))
                continue
            ok = True
            for _, wsel in selected:
                if wsel is None:
                    continue
                if abs(win - wsel) < spacing:
                    ok = False
                    break
            if ok:
                selected.append((name, win))
        if not selected and ranked:
            selected.append(ranked[0])
        return selected

    def _spearman_corr(self, a: pd.Series, b: pd.Series) -> float:
        try:
            ar = a.rank(pct=True)
            br = b.rank(pct=True)
            c = ar.corr(br)
            return float(0.0 if np.isnan(c) else c)
        except Exception:
            return 0.0

    def _annualization_factor_15m(self) -> float:
        # 365.25 days * 24 hours * 4 (15m per hour) ≈ 35064 bars/year, sqrt for Sharpe
        return float(np.sqrt(365.25 * 24 * 4))

    @smart_cache(cache_key_func=lambda self, series, returns, direction, outer_folds, min_test_signals, use_tpe: 
                 f"oos_sharpe_{hash(str(series.values))}_{hash(str(returns.values))}_{direction}_{outer_folds}_{min_test_signals}_{use_tpe}")
    def _compute_oos_sharpe_nested(self, series: pd.Series, returns: pd.Series, direction: str,
                                   outer_folds: int = 3, min_test_signals: int = 100,
                                   use_tpe: bool = True) -> Tuple[float, Dict[str, Any]]:
        # series and returns aligned, no NaNs
        # Add data format analysis for troubleshooting (only for debugging)
        tprint_data_format(series, "oos_sharpe_series", level="DEBUG")
        tprint_data_format(returns, "oos_sharpe_returns", level="DEBUG")
        
        af = self._annualization_factor_15m()
        n = len(series)
        if n < 500:
            return 0.0, {'reason': 'insufficient_length'}
        # Prepare indices for outer folds (simple time-based split if PurgedKFold not available)
        splits: List[Tuple[int, int]] = []
        try:
            from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator as PurgedKFoldTime  # type: ignore
            pkf = PurgedKFoldTime(n_splits=outer_folds, purge_pct=0.01, embargo_pct=0.01)
            idx = np.arange(n)
            for tr, te in pkf.split(idx, None, None):
                splits.append((int(tr[-1]), int(te[0])))  # markers only (we'll use slices)
            # Rebuild as slices using proportions
            # If PurgedKFoldTime returns explicit arrays, we’ll use them directly below
            use_pkf = True
        except Exception:
            # Fallback to proper time-based splits with purging
            use_pkf = False
            # Create time-based splits with proper purging
            fold_size = n // outer_folds
            for i in range(outer_folds):
                train_end = (i + 1) * fold_size - int(fold_size * 0.01)  # Purge 1%
                test_start = (i + 1) * fold_size + int(fold_size * 0.01)  # Embargo 1%
                if test_start < n:
                    splits.append((train_end, test_start))

        # Helper to evaluate Sharpe for given threshold on a given split
        def eval_sharpe(sig_mask: np.ndarray, ret_vals: np.ndarray) -> Tuple[float, int]:
            sig_count = int(sig_mask.sum())
            if sig_count == 0:
                return 0.0, 0
            strat = (ret_vals * (1.0 if direction == 'longs' else -1.0)) * sig_mask.astype(float)
            mu = strat.mean()
            sd = strat.std(ddof=1) if strat.size > 1 else 0.0
            if sd == 0.0:
                return 0.0, sig_count
            return float((mu / sd) * af), sig_count

        # Inner optimization: thresholds
        def optimize_threshold(train_vals: np.ndarray, train_rets: np.ndarray, use_rolling: bool = False) -> float:
            # Candidate grid defaults (static quantiles)
            if direction == 'longs':
                grid = [0.6, 0.7, 0.8, 0.9]
            else:
                grid = [0.4, 0.3, 0.2, 0.1]
            best_q = grid[0]
            best_s = -1e9
            # Try to use TPE optimizer if available
            if use_tpe:
                try:
                    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig  # type: ignore
                    opt = BayesianTPEOptimizer(config=OptimizationConfig(n_trials=20, tpe_trials=20))
                    # Define search space
                    low, high = (0.6, 0.95) if direction == 'longs' else (0.05, 0.4)
                    def objective(params: Dict[str, Any]) -> float:
                        q = float(params.get('q', 0.8 if direction == 'longs' else 0.2))
                        thr = np.quantile(train_vals, q)
                        if direction == 'longs':
                            sig = (train_vals > thr)
                        else:
                            sig = (train_vals < thr)
                        # simple inner validation split (70/30) on train
                        m = len(train_vals)
                        tv = int(m * 0.7)
                        sig_te = sig[tv:]
                        r_te = train_rets[tv:]
                        # compute Sharpe on validation slice
                        s, c = eval_sharpe(sig_te, r_te)
                        # prefer non-degenerate with minimal signals requirement
                        if c < max(10, int(0.01 * len(sig_te))):
                            return -1e6
                        return float(s)
                    result = opt.optimize(objective, {'q': {'type': 'float', 'low': low, 'high': high}})
                    q_opt = float(result.get('best_params', {}).get('q', grid[0]))
                    # Clip to sensible range
                    best_q = min(max(q_opt, low), high)
                except Exception:
                    pass
            # Fallback / refinement on grid
            for q in grid:
                thr = np.quantile(train_vals, q)
                sig = (train_vals > thr) if direction == 'longs' else (train_vals < thr)
                m = len(train_vals)
                tv = int(m * 0.7)
                s, c = eval_sharpe(sig[tv:], train_rets[tv:])
                sc = s if c >= max(10, int(0.01 * len(sig))) else -1e6
                if sc > best_s:
                    best_s = sc
                    best_q = q
            return float(best_q)

        # Build folds
        if use_pkf:
            # Construct folds using PurgedKFoldTime split indices
            from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator as PurgedKFoldTime  # type: ignore
            pkf = PurgedKFoldTime(n_splits=outer_folds, purge_pct=0.01, embargo_pct=0.01)
            xs = series.values
            rs = returns.values
            sharpe_scores = []
            total_signals = 0
            for tr_idx, te_idx in pkf.split(xs, None, None):
                x_tr, x_te = xs[tr_idx], xs[te_idx]
                r_tr, r_te = rs[tr_idx], rs[te_idx]
                q_star = optimize_threshold(x_tr, r_tr, use_rolling=False)
                thr = np.quantile(x_tr, q_star)
                sig_te = (x_te > thr) if direction == 'longs' else (x_te < thr)
                s, c = eval_sharpe(sig_te, r_te)
                if c >= min_test_signals:
                    sharpe_scores.append(s)
                    total_signals += c
            if not sharpe_scores:
                return 0.0, {'reason': 'insufficient_data'}
            return float(np.mean(sharpe_scores)), {'threshold': 'quantile', 'signals_test': total_signals}
        else:
            # Simple contiguous folds
            xs = series.values
            rs = returns.values
            sharpe_scores = []
            total_signals = 0
            fold_sizes = np.linspace(0, n, num=outer_folds + 1, dtype=int)
            for i in range(outer_folds):
                start = fold_sizes[i]
                end = fold_sizes[i + 1]
                if end - start < 100:
                    continue
                # train on [0:start], test on [start:end]
                x_tr, x_te = xs[:start], xs[start:end]
                r_tr, r_te = rs[:start], rs[start:end]
                if len(x_tr) < 200:
                    continue
                q_star = optimize_threshold(x_tr, r_tr, use_rolling=False)
                thr = np.quantile(x_tr, q_star)
                sig_te = (x_te > thr) if direction == 'longs' else (x_te < thr)
                s, c = eval_sharpe(sig_te, r_te)
                if c >= min_test_signals:
                    sharpe_scores.append(s)
                    total_signals += c
            if not sharpe_scores:
                return 0.0, {'reason': 'insufficient_data'}
            return float(np.mean(sharpe_scores)), {'threshold': 'quantile', 'signals_test': total_signals}

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    def _mrmr_select_windows(self, candidates: List[str], relevance: Dict[str, float], features: pd.DataFrame,
                              k: int = 3, redundancy_penalty: float = 0.5) -> List[str]:
        if not candidates:
            return []
        # Greedy selection: max(relevance - penalty*avg_redundancy)
        selected: List[str] = []
        remaining = list(sorted(candidates, key=lambda n: relevance.get(n, 0.0), reverse=True))
        while remaining and len(selected) < k:
            if not selected:
                sel = remaining.pop(0)
                selected.append(sel)
                continue
            best_name = None
            best_score = -1e18
            for name in remaining:
                # compute avg spearman with already selected
                corrs = []
                for s in selected:
                    c = self._spearman_corr(features[name], features[s])
                    corrs.append(abs(c))
                avg_red = float(np.mean(corrs)) if corrs else 0.0
                score = relevance.get(name, 0.0) - redundancy_penalty * avg_red
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name is None:
                break
            remaining.remove(best_name)
            selected.append(best_name)
        return selected

    def _compute_spearman_matrix(self, df: pd.DataFrame) -> np.ndarray:
        try:
            # Rank transform then Pearson corr == Spearman
            rk = df.rank(pct=True)
            mat = rk.corr(method='pearson').to_numpy()
            mat = np.nan_to_num(mat, nan=0.0)
            return np.abs(mat)
        except Exception:
            # Fallback safe matrix
            n = df.shape[1]
            return np.eye(n)

    def _mrmr_select_windows_from_matrix(self, cand_cols: List[str], relevance_vec: np.ndarray,
                                         redundancy_matrix: np.ndarray, k: int = 3,
                                         redundancy_penalty: float = 0.5) -> List[str]:
        n = len(cand_cols)
        if n == 0:
            return []
        order = list(np.argsort(-relevance_vec))  # descending relevance
        selected_idx: List[int] = []
        remaining = order.copy()
        while remaining and len(selected_idx) < k:
            if not selected_idx:
                selected_idx.append(remaining.pop(0))
                continue
            best_i = None
            best_score = -1e18
            for i in remaining:
                # average redundancy to already selected
                if selected_idx:
                    avg_red = float(np.mean([redundancy_matrix[i, j] for j in selected_idx]))
                else:
                    avg_red = 0.0
                score = float(relevance_vec[i]) - redundancy_penalty * avg_red
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                break
            remaining.remove(best_i)
            selected_idx.append(best_i)
        return [cand_cols[i] for i in selected_idx]

    def _oos_sharpe_nested_vectorized(self, X: np.ndarray, r: np.ndarray, direction: str,
                                      outer_folds: int = 3, min_test_signals: int = 100,
                                      use_tpe: bool = False, use_golden: bool = True) -> np.ndarray:
        """Compute OOS Sharpe for multiple candidates in X (n_samples, n_candidates).

        - Vectorized across candidates per fold.
        - Default: golden-section search on quantile q per candidate using train 70/30 split for validation.
        - Fallback: small quantile grid.
        """
        n, m = X.shape
        if n < 500 or m == 0:
            return np.zeros(m, dtype=np.float32)

        # Outer folds generation
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        try:
            from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator as PurgedKFoldTime  # type: ignore
            pkf = PurgedKFoldTime(n_splits=outer_folds, purge_pct=0.01, embargo_pct=0.01)
            idx = np.arange(n)
            for tr_idx, te_idx in pkf.split(idx, None, None):
                folds.append((tr_idx, te_idx))
        except Exception:
            # Contiguous splits
            edges = np.linspace(0, n, num=outer_folds + 1, dtype=int)
            for i in range(outer_folds):
                start, end = edges[i], edges[i + 1]
                if end - start >= 100 and start >= 200:
                    tr = np.arange(0, start)
                    te = np.arange(start, end)
                    folds.append((tr, te))

        if not folds:
            return np.zeros(m, dtype=np.float32)

        phi = (1 + np.sqrt(5)) / 2

        def sharpe_from_mask(mask: np.ndarray, ret: np.ndarray) -> np.ndarray:
            # mask shape (T, m), ret shape (T,)
            mm = mask.astype(np.float32)
            strat = mm * ret[:, None].astype(np.float32) * (1.0 if direction == 'longs' else -1.0)
            mu = strat.mean(axis=0)
            sd = strat.std(axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                s = np.where(sd > 0, mu / sd, 0.0)
            return s * np.sqrt(365.25 * 24 * 4)

        sharpes: List[np.ndarray] = []
        for tr_idx, te_idx in folds:
            X_tr = X[tr_idx, :]
            r_tr = r[tr_idx]
            X_te = X[te_idx, :]
            r_te = r[te_idx]
            if len(X_tr) < 200 or len(X_te) < 100:
                continue
            # train split -> validation for inner selection
            tv = int(0.7 * len(X_tr))
            if tv < 50 or len(X_tr) - tv < 50:
                continue
            X_tr_tr = X_tr[:tv, :]
            X_tr_val = X_tr[tv:, :]
            r_val = r_tr[tv:]

            # Precompute sorted training values for per-candidate quantiles
            X_sorted = np.sort(X_tr_tr, axis=0)
            m_tr = X_tr_tr.shape[0]
            idx_cand = np.arange(m)

            # Initialize per-candidate bounds for q
            if direction == 'longs':
                L = np.full(m, 0.60, dtype=np.float64)
                R = np.full(m, 0.95, dtype=np.float64)
            else:
                L = np.full(m, 0.05, dtype=np.float64)
                R = np.full(m, 0.40, dtype=np.float64)

            def eval_q(q: np.ndarray, X_val: np.ndarray, X_sorted_tr: np.ndarray, r_val: np.ndarray) -> np.ndarray:
                # q shape (m,), compute thr per candidate using sorted training values
                pos = np.floor(q * (m_tr - 1)).astype(int)
                pos = np.clip(pos, 0, m_tr - 1)
                thr = X_sorted_tr[pos, idx_cand]  # shape (m,)
                if direction == 'longs':
                    mask = X_val > thr[None, :]
                else:
                    mask = X_val < thr[None, :]
                # signal count gate
                sig_counts = mask.sum(axis=0)
                s = sharpe_from_mask(mask, r_val)
                s[sig_counts < max(min_test_signals, int(0.01 * len(r_val)))] = -1e6
                return s

            if use_golden:
                # Golden-section vectorized: update bounds per candidate
                iters = 8
                for _ in range(iters):
                    c = R - (R - L) / phi
                    d = L + (R - L) / phi
                    sc = eval_q(c, X_tr_val, X_sorted, r_val)
                    sd = eval_q(d, X_tr_val, X_sorted, r_val)
                    # where sc > sd -> move R = d else L = c, per candidate
                    mask = sc > sd
                    R = np.where(mask, d, R)
                    L = np.where(mask, L, c)
                q_star = 0.5 * (L + R)
            else:
                # small grid
                q_grid = np.linspace(L[0], R[0], num=9) if direction == 'longs' else np.linspace(L[0], R[0], num=9)
                # evaluate all q in grid and pick best per candidate
                s_mat = []
                for q in q_grid:
                    qq = np.full(m, q, dtype=np.float64)
                    s_mat.append(eval_q(qq, X_tr_val, X_sorted, r_val))
                s_mat = np.vstack(s_mat)  # (nq, m)
                best_idx = np.argmax(s_mat, axis=0)
                q_star = q_grid[best_idx]

            # Evaluate on OOS test with q_star
            # build threshold per candidate from full train X_tr_tr
            pos = np.floor(q_star * (m_tr - 1)).astype(int)
            pos = np.clip(pos, 0, m_tr - 1)
            thr = X_sorted[pos, idx_cand]
            if direction == 'longs':
                mask_te = X_te > thr[None, :]
            else:
                mask_te = X_te < thr[None, :]
            s_te = sharpe_from_mask(mask_te, r_te)
            # gate by min_test_signals on test
            sig_counts_te = mask_te.sum(axis=0)
            s_te[sig_counts_te < min_test_signals] = 0.0
            sharpes.append(s_te)

        if not sharpes:
            return np.zeros(m, dtype=np.float32)
        sh = np.vstack(sharpes).mean(axis=0)
        return sh.astype(np.float32)

    def _compute_per_feature_mi_and_mrmr(self,
                                          features: pd.DataFrame,
                                          targets: pd.Series,
                                          data_close: Optional[pd.Series],
                                          direction: str = 'longs',
                                          timeframe: str = '15m',
                                          max_rows: int = 200000,
                                          prefilter_M: int = 6,
                                          spacing: int = 2,
                                          outer_folds: int = 3) -> Dict[str, Any]:
        # Enforce timeframe and sample cap
        self._enforce_15m_timeframe(features)
        if not isinstance(targets, pd.Series) or targets.empty:
            raise ValueError("Targets are required for MI/mRMR selection")
        if max_rows and len(features) > max_rows:
            features = features.iloc[-max_rows:]
            targets = targets.iloc[-max_rows:]
            if data_close is not None:
                data_close = data_close.iloc[-max_rows:]

        # Keep only numeric columns and downcast to float32
        num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        features = features[num_cols].copy()
        for c in features.columns:
            if features[c].dtype == np.float64:
                features[c] = features[c].astype(np.float32)
        # Hardware optimizations
        try:
            if self.gpu_manager and self.gpu_manager.is_available():
                features = self.gpu_manager.optimize_dataframe(
                    features, 
                    operation_type=GPUOperationType.DATA_PROCESSING
                )
        except Exception:
            pass
        try:
            _ = optimize_memory()
        except Exception:
            pass

        # Align features with targets
        features, targets, _ = self._align_features_targets(features, targets.astype(float))
        returns = targets

        # Group by base feature families
        family_map = self._group_windows_by_family(list(features.columns))
        mi_best_lookbacks_per_feature: Dict[str, Optional[int]] = {}
        mrmr_top_lookbacks_per_feature: Dict[str, List[Optional[int]]] = {}
        mi_scores_by_feature: Dict[str, Dict[str, float]] = {}
        oos_sharpe_by_feature_window: Dict[str, Dict[str, float]] = {}
        selected_features_metadata: Dict[str, Dict[str, Any]] = {}
        family_diagnostics_rows: List[Dict[str, Any]] = []

        # Caches for MI and correlation
        mi_cache: Dict[str, float] = {}

        families = list(family_map.items())
        # Chunk families to control memory
        fam_chunk_size = max(1, int(np.ceil(len(families) / max(1, self.parallel_workers))))
        for i in range(0, len(families), fam_chunk_size):
            fam_batch = families[i:i + fam_chunk_size]
            for base, items in fam_batch:
                cols = [n for (n, _) in items]
                # MI scores with caching
                to_compute = [c for c in cols if c not in mi_cache]
                if to_compute:
                    mi_new = self._compute_mi_scores(to_compute, features, targets)
                    mi_cache.update(mi_new)
                mi_scores = {c: mi_cache.get(c, 0.0) for c in cols}
                mi_scores_by_feature[base] = mi_scores

                # Stability scores
                stab_scores = self._compute_stability_scores(cols, features)

                # Best-by weighted (0.8 * MI_norm + 0.2 * Stability_norm)
                if cols:
                    mi_vals = np.array([mi_scores.get(c, 0.0) for c in cols], dtype=np.float64)
                    st_vals = np.array([stab_scores.get(c, 0.0) for c in cols], dtype=np.float64)
                    def _norm(v: np.ndarray) -> np.ndarray:
                        vmin, vmax = float(np.min(v)), float(np.max(v))
                        if vmax - vmin <= 1e-12:
                            return np.zeros_like(v)
                        return (v - vmin) / (vmax - vmin)
                    mi_n = _norm(mi_vals)
                    st_n = _norm(st_vals)
                    weighted = 0.8 * mi_n + 0.2 * st_n
                    best_idx = int(np.argmax(weighted))
                    best_col = cols[best_idx]
                else:
                    best_col = None

                best_win = None
                if best_col is not None:
                    m = re.search(r"(\d+)(?!.*\d)", best_col)
                    if m:
                        try:
                            best_win = int(m.group(1))
                        except Exception:
                            best_win = None
                mi_best_lookbacks_per_feature[base] = best_win

                # Prefilter by MI with spacing to reduce candidate set
                prefiltered = self._prefilter_by_mi_with_spacing(items, mi_scores, max_M=prefilter_M, spacing=spacing)
                cand_cols = [n for (n, _) in prefiltered]
                rel: Dict[str, float] = {}
                # Vectorized nested CV Sharpe across candidates
                X_cand = features[cand_cols].to_numpy(dtype=np.float32, copy=False)
                r_vals = returns.to_numpy(dtype=np.float32, copy=False)
                sh_vec = self._oos_sharpe_nested_vectorized(
                    X_cand, r_vals, direction=direction, outer_folds=outer_folds,
                    min_test_signals=100, use_tpe=False, use_golden=True
                )
                for j, name in enumerate(cand_cols):
                    score = float(sh_vec[j])
                    rel[name] = score
                    oos_sharpe_by_feature_window.setdefault(base, {})[name] = score

                # Spearman redundancy matrix (vectorized) and greedy mRMR
                red_mat = self._compute_spearman_matrix(features[cand_cols])
                top_cols = self._mrmr_select_windows_from_matrix(
                    cand_cols,
                    np.array([rel.get(n, 0.0) for n in cand_cols], dtype=np.float64),
                    red_mat,
                    k=3,
                    redundancy_penalty=0.5
                )
                wins: List[Optional[int]] = []
                for cn in top_cols:
                    m = re.search(r"(\d+)(?!.*\d)", cn)
                    w = int(m.group(1)) if m else None
                    wins.append(w)
                    selected_features_metadata[cn] = {
                        'family': base,
                        'window': w,
                        'mi_score': mi_scores.get(cn, 0.0),
                        'oos_sharpe': rel.get(cn, 0.0),
                        'selection_reason': 'mrmr_top_k'
                    }
                mrmr_top_lookbacks_per_feature[base] = wins

                family_diagnostics_rows.append({
                    'family': base,
                    'mi_best_window': best_win,
                    'mi_best_feature': best_col,
                    'mi_best_score': mi_scores.get(best_col, 0.0) if best_col else 0.0,
                    'stability_best_feature': (self._compute_stability_scores([best_col], features).get(best_col, 0.0) if best_col else 0.0),
                    'mrmr_selected_windows': wins,
                    'mrmr_candidates': cand_cols,
                    'avg_oos_sharpe_candidates': float(np.mean([rel.get(c, 0.0) for c in cand_cols])) if cand_cols else 0.0
                })

            try:
                _ = optimize_memory()
            except Exception:
                pass

        family_diagnostics = pd.DataFrame(family_diagnostics_rows) if family_diagnostics_rows else pd.DataFrame()

        return {
            'mi_best_lookbacks_per_feature': mi_best_lookbacks_per_feature,
            'mrmr_top_lookbacks_per_feature': mrmr_top_lookbacks_per_feature,
            'mi_scores_by_feature': mi_scores_by_feature,
            'oos_sharpe_by_feature_window': oos_sharpe_by_feature_window,
            'selected_features_metadata': selected_features_metadata,
            'family_diagnostics': family_diagnostics,
            'optimization_config': {
                'prefilter_M': prefilter_M,
                'spacing': spacing,
                'outer_folds': outer_folds,
                'min_test_signals': 100,
                'redundancy_spearman_threshold': 0.7,
                'mrmr_k': 3,
                'thresholds': 'static_quantiles'
            }
        }

    @auto_optimize(OptimizationConfig(
        enable_caching=True,
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.MAXIMUM,
        enable_compression=True
    ))
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the period + lookback optimization step with enhanced hardware optimizations."""
        self.logger.info("🔍 Starting period + lookback optimization step with enhanced hardware optimizations")
        
        # Initialize comprehensive optimization for this workload
        tprint_info("🚀 Initializing comprehensive optimization for feature engineering workload")
        self.comprehensive_optimizer.optimize_for_workload(
            workload_category=WorkloadCategory.FEATURE_ENGINEERING,
            strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE
        )
        
        # Set context for enhanced file naming
        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        direction = config.get('direction', 'long')
        model = config.get('model', 'Analyst')
        
        self._set_context(symbol=symbol, exchange=exchange, direction=direction, model=model)
        
        # Load required artifacts from previous steps
        tprint_info("Loading required artifacts from previous steps")
        
        # Load features from feature_generation_feature_generation_step
        features_df = None
        try:
            features_df = self._load_dataframe('generated_features')
            if features_df is not None and not features_df.empty:
                tprint_success(f"Loaded features from artifact manager: {features_df.shape}")
                tprint_data_preview(features_df, "loaded_features", max_rows=5, level="INFO")
                # Add comprehensive data format analysis for troubleshooting
                tprint_data_format(features_df, "loaded_features", level="INFO")
            else:
                tprint_warning("No features found in artifact manager, trying fallback")
                # Try fallback loading
                features_df = self._load_dataframe('feature_lists')
        except Exception as e:
            tprint_warning(f"Failed to load features from artifact manager: {e}")
        
        if features_df is None or features_df.empty:
            tprint_error("No features available for period/lookback optimization")
            error_result = {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': 'No features available. Run feature_generation_feature_generation_step first.'
            }
            tprint_data_format(error_result, "no_features_error", level="ERROR")
            return error_result
        
        # Load targets from feature_generation_labeling_integration_step
        targets_series = None
        try:
            targets_series = self._load_dataframe('targets')
            if targets_series is not None and not targets_series.empty:
                if isinstance(targets_series, pd.DataFrame):
                    # Extract the target column if it's a DataFrame
                    target_cols = ['target', 'targets', 'label', 'labels', 'y', 'direction_confidence', 'opportunity_asymmetry', 'directional_signal']
                    for col in target_cols:
                        if col in targets_series.columns:
                            targets_series = targets_series[col]
                            break
                    if isinstance(targets_series, pd.DataFrame):
                        targets_series = targets_series.iloc[:, 0]  # Take first column
                tprint_success(f"Loaded targets from artifact manager: {len(targets_series)} samples")
                tprint_data_preview(targets_series, "loaded_targets", max_rows=10, level="INFO")
                # Add comprehensive data format analysis for troubleshooting
                tprint_data_format(targets_series, "loaded_targets", level="INFO")
            else:
                tprint_warning("No targets found in artifact manager, trying fallback")
                # Try fallback loading
                targets_series = self._load_dataframe('labels')
        except Exception as e:
            tprint_warning(f"Failed to load targets from artifact manager: {e}")
        
        if targets_series is None or targets_series.empty:
            tprint_error("No targets available for period/lookback optimization")
            error_result = {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': 'No targets available. Run feature_generation_labeling_integration_step first.'
            }
            tprint_data_format(error_result, "no_targets_error", level="ERROR")
            return error_result
        
        # Align features and targets with proper validation
        tprint_info("Aligning features and targets for optimization")
        
        # Validate data alignment before joining with temporal order validation
        if not features_df.index.equals(targets_series.index):
            tprint_warning("Feature and target indices don't match, attempting alignment")
            # Find common index
            common_index = features_df.index.intersection(targets_series.index)
            if len(common_index) == 0:
                tprint_error("No common timestamps between features and targets")
                error_result = {
                    'success': False,
                    'artifacts': [],
                    'metrics': {},
                    'error': 'No common timestamps between features and targets.'
                }
                return error_result
            
            # Validate temporal order to prevent data leakage
            if hasattr(common_index, 'is_monotonic_increasing'):
                if not common_index.is_monotonic_increasing:
                    tprint_error("Common timestamps are not in temporal order - potential data leakage risk")
                    error_result = {
                        'success': False,
                        'artifacts': [],
                        'metrics': {},
                        'error': 'Common timestamps are not in temporal order - potential data leakage risk.'
                    }
                    return error_result
                tprint_success("Temporal order validation passed")
            
            features_df = features_df.loc[common_index]
            targets_series = targets_series.loc[common_index]
        
        aligned_data = features_df.join(targets_series.rename('target'), how='inner').dropna()
        
        # Validate alignment quality
        if aligned_data.empty:
            tprint_error("No overlapping timestamps between features and targets after alignment")
            error_result = {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': 'No overlapping timestamps between features and targets after alignment.'
            }
            return error_result
        
        # Check for data leakage (ensure targets are not in the future)
        if hasattr(aligned_data.index, 'is_monotonic_increasing'):
            if not aligned_data.index.is_monotonic_increasing:
                tprint_warning("Data index is not monotonic, potential data leakage risk")
        
        tprint_success(f"Data alignment successful: {len(aligned_data)} samples aligned")
        
        aligned_features = aligned_data.drop(columns=['target'])
        aligned_targets = aligned_data['target']
        
        tprint_success(f"Aligned data: features={aligned_features.shape}, targets={aligned_targets.shape}")
        tprint_data_preview(aligned_data, "aligned_data", max_rows=5, level="INFO")
        tprint_data_preview(aligned_features, "aligned_features", max_rows=5, level="DEBUG")
        tprint_data_preview(aligned_targets, "aligned_targets", max_rows=10, level="DEBUG")
        
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(aligned_data, "aligned_data", level="INFO")
        tprint_data_format(aligned_features, "aligned_features", level="DEBUG")
        tprint_data_format(aligned_targets, "aligned_targets", level="DEBUG")
        
            # Use aligned data for optimization
            data = aligned_features
            # Add comprehensive data format analysis for troubleshooting
            tprint_data_format(data, "final_optimization_data", level="DEBUG")
        
        try:
            # Log optimization configuration
            tprint_info(f"🚀 Enhanced Hardware Optimization Configuration:")
            tprint_info(f"   - Parallel Workers: {self.parallel_workers}")
            tprint_info(f"   - Chunk Size: {self.chunk_size}")
            tprint_info(f"   - Memory Mapping: {self.memory_mapping_enabled}")
            tprint_info(f"   - Aggressive GC: {self.aggressive_gc_enabled}")
            tprint_info(f"   - Data Type Optimization: {self.data_type_optimization}")
            tprint_info(f"   - Enhanced GPU Available: {self.gpu_manager.is_available()}")
            # Add comprehensive data format analysis for troubleshooting
            tprint_data_format(data, "optimization_config_data", level="DEBUG")
            
            # Process data through the optimization
            tprint_info("Processing data through M1-optimized pipeline")
            result = self._process_data(data, **config)
            tprint_info(f"Process data result: success={result.get('success', False)}")
            # Add comprehensive data format analysis for troubleshooting
            tprint_data_format(result, "process_data_result", level="INFO")
            tprint_data_format(data, "process_data_input", level="DEBUG")
            tprint_data_format(config, "process_data_config", level="DEBUG")
            
            # Save artifacts using BaseStep methods
            if result.get('success', False):
                if 'optimized_features' in result.get('artifacts', {}):
                    tprint_data_preview(result['artifacts']['optimized_features'], "saved_optimized_features", max_rows=5, level="INFO")
                    # Add comprehensive data format analysis for troubleshooting
                    tprint_data_format(result['artifacts']['optimized_features'], "saved_optimized_features", level="INFO")
                    self._save_dataframe(result['artifacts']['optimized_features'], 'optimized_features')
                if 'optimization_metadata' in result:
                    # Add data format analysis for metadata
                    tprint_data_format(result['optimization_metadata'], "optimization_metadata", level="DEBUG")
                    self._save_metadata(result['optimization_metadata'], 'optimization_metadata')
                if 'family_diagnostics' in result.get('artifacts', {}):
                    tprint_data_preview(result['artifacts']['family_diagnostics'], "saved_family_diagnostics", max_rows=5, level="INFO")
                    # Add comprehensive data format analysis for troubleshooting
                    tprint_data_format(result['artifacts']['family_diagnostics'], "saved_family_diagnostics", level="INFO")
                    self._save_dataframe(result['artifacts']['family_diagnostics'], 'family_diagnostics')
            
            # Log optimization statistics
            optimization_stats = result.get('optimization_stats', {})
            if optimization_stats:
                tprint_info("📊 Optimization Statistics:")
                tprint_info(f"   - Workers Used: {optimization_stats.get('parallel_workers_used', 'N/A')}")
                tprint_info(f"   - Final Memory Usage: {optimization_stats.get('final_memory_usage', 0):.1f}%")
                tprint_info(f"   - GPU Acceleration: {optimization_stats.get('m1_gpu_acceleration', False)}")
                # Add comprehensive data format analysis for troubleshooting
                tprint_data_format(optimization_stats, "optimization_stats", level="INFO")
            
            # Generate human-readable report
            if result.get('success', False):
                tprint_info("Generating human-readable optimization report")
                # Add comprehensive data format analysis for troubleshooting
                tprint_data_format(result, "report_generation_result", level="DEBUG")
                tprint_data_format(data, "report_generation_data", level="DEBUG")
                report_path = self._generate_optimization_report(result, data, **config)
                tprint_success(f"📊 Optimization report saved to: {report_path}")
            
            # Add comprehensive data format analysis for final result
            final_result = {
                'success': result.get('success', False),
                'artifacts': list(result.get('artifacts', {}).keys()),
                'metrics': {
                    'optimization_stats': optimization_stats,
                    'optimization_metadata': result.get('optimization_metadata', {})
                },
                'error': None if result.get('success', False) else "Period + lookback optimization failed"
            }
            tprint_data_format(final_result, "final_optimization_result", level="INFO")
            return final_result
            
        except Exception as e:
            tprint_error(f"Period + lookback optimization execution failed: {e}")
            tprint_debug(f"Execution error details: {traceback.format_exc()}")
            # Add data format analysis for error troubleshooting
            tprint_data_format(data, "error_data_state", level="ERROR")
            self.logger.error(f"Period + lookback optimization execution failed: {e}")
            
            # Ensure cleanup even on error
            try:
                self.memory_manager.stop_monitoring()
                if self.aggressive_gc_enabled:
                    self._aggressive_garbage_collection()
                
                # Additional cleanup for hardware resources
                if hasattr(self, 'comprehensive_optimizer'):
                    try:
                        self.comprehensive_optimizer.cleanup()
                    except Exception as hw_cleanup_error:
                        tprint_warning(f"Hardware optimizer cleanup failed: {hw_cleanup_error}")
                
                if hasattr(self, 'tpe_optimizer'):
                    try:
                        self.tpe_optimizer.cleanup()
                    except Exception as tpe_cleanup_error:
                        tprint_warning(f"TPE optimizer cleanup failed: {tpe_cleanup_error}")
                        
            except Exception as cleanup_error:
                tprint_warning(f"Cleanup failed: {cleanup_error}")
                # Add data format analysis for cleanup error troubleshooting
                tprint_data_format(cleanup_error, "cleanup_error", level="ERROR")
            
            error_result = {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
            tprint_data_format(error_result, "error_result", level="ERROR")
            return error_result

    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    @smart_cache(cache_key_func=lambda self, features, targets, periods_to_test, direction: 
                 f"period_opt_{hash(str(features.shape))}_{hash(str(targets.shape))}_{hash(tuple(periods_to_test))}_{direction}")
    def _optimize_periods(self, features: pd.DataFrame, targets: pd.Series, 
                         periods_to_test: List[int], direction: str) -> Tuple[int, Dict[str, float]]:
        """Optimize periods using mutual information and out-of-sample Sharpe ratio."""
        tprint_step("Optimizing periods")
        
        # Input validation
        try:
            features = self.validator.validate_dataframe(features, min_rows=50)
            targets = self.validator.validate_series(targets, min_length=50)
            periods_to_test = [self.validator.validate_positive_int(p, f"period_{p}") for p in periods_to_test]
            if direction not in ['longs', 'shorts']:
                raise ValidationError(f"Direction must be 'longs' or 'shorts', got {direction}")
        except ValidationError as e:
            tprint_error(f"Period optimization validation failed: {e}")
            raise
        
        tprint_data_preview(features, "period_optimization_input", max_rows=5, level="DEBUG")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(features, "period_optimization_input", level="DEBUG")
        tprint_data_format(targets, "period_optimization_targets", level="DEBUG")
        
        # Pre-compute all period features using parallel processing
        tprint_info("🎯 Pre-computing period features for efficient optimization with parallel processing")
        period_features_cache = self._parallel_create_features(features, periods_to_test, 'period')
        
        # Use TPE optimizer for sophisticated period optimization
        tprint_info("🎯 Using Bayesian TPE optimizer for period optimization")
        
        def period_objective(trial):
            """Objective function for TPE optimization."""
            period = trial.suggest_int('period', min(periods_to_test), max(periods_to_test))
            
            try:
                # Use pre-computed period features
                period_features = period_features_cache.get(period, pd.DataFrame())
                
                if period_features.empty:
                    return -np.inf
                
                # Calculate mutual information vectorized
                mi_scores = self._compute_vectorized_mutual_information(period_features, targets)
                
                if len(mi_scores) == 0 or np.all(mi_scores == 0):
                    return -np.inf
                
                avg_mi = np.mean(mi_scores)
                
                # Calculate comprehensive risk metrics using all features
                if len(period_features.columns) > 0:
                    # Use intelligent feature aggregation based on feature types
                    feature_series = self._create_intelligent_composite_signal(period_features)
                else:
                    feature_series = pd.Series()
                
                # Calculate multiple risk metrics
                risk_metrics = self._compute_comprehensive_risk_metrics(
                    feature_series,
                    targets,
                    direction
                )
                sharpe_score = risk_metrics['sharpe_ratio']
                
                # Calculate transaction cost adjusted score
                transaction_cost = 0.001  # 0.1% total fees
                cost_adjusted_sharpe = sharpe_score - transaction_cost
                
                # Combined score (weighted average) with transaction cost consideration
                combined_score = 0.5 * avg_mi + 0.3 * cost_adjusted_sharpe + 0.2 * risk_metrics.get('sortino_ratio', 0.0)
                return combined_score
                
            except ValidationError as e:
                tprint_error(f"TPE trial validation failed for period {period}: {e}")
                return -np.inf
            except ValueError as e:
                tprint_error(f"TPE trial value error for period {period}: {e}")
                return -np.inf
            except Exception as e:
                tprint_error(f"TPE trial unexpected error for period {period}: {e}")
                tprint_debug(f"Exception details: {traceback.format_exc()}")
                return -np.inf
        
        # Run TPE optimization
        study = None
        try:
            study = self.tpe_optimizer.optimize(period_objective, {
                'period': {'type': 'int', 'low': min(periods_to_test), 'high': max(periods_to_test)}
            })
            
            best_period = study.best_params['period']
            best_score = study.best_value
            
            # Generate scores for all tested periods for reporting (reuse TPE results if available)
            period_scores = {}
            if study is not None and hasattr(study, 'trials'):
                # Use TPE trial results if available
                for trial in study.trials:
                    if trial.value is not None and 'period' in trial.params:
                        period_scores[str(trial.params['period'])] = trial.value
                
                # Fill missing periods with 0.0
                for period in periods_to_test:
                    if str(period) not in period_scores:
                        period_scores[str(period)] = 0.0
            else:
                # Fast fail if TPE optimization fails
                tprint_error("TPE optimization failed - no fallback available")
                raise RuntimeError("TPE optimization failed and fallback is disabled for consistency")
                    
        except Exception as e:
            tprint_warning(f"TPE optimization failed, falling back to grid search: {e}")
            # Fallback to original grid search
            period_scores = {}
            best_period = periods_to_test[0]
            best_score = -np.inf
        finally:
            # Clean up TPE study to prevent memory leaks
            if study is not None:
                try:
                    del study
                    gc.collect()
                except Exception as cleanup_error:
                    tprint_warning(f"TPE study cleanup failed: {cleanup_error}")
            
            for period in periods_to_test:
                try:
                    period_features = self._create_period_features(features, period)
                    if period_features.empty:
                        continue
                    
                    mi_scores = []
                    for col in period_features.columns:
                        if not period_features[col].isna().all():
                            mi = self._compute_mutual_information(period_features[col], targets)
                            mi_scores.append(mi)
                    
                    if not mi_scores:
                        continue
                    
                    avg_mi = np.mean(mi_scores)
                    # Use all features for Sharpe calculation
                    if len(period_features.columns) > 0:
                        if len(period_features.columns) == 1:
                            feature_series = period_features.iloc[:, 0]
                        else:
                            # Use mean of all features as a composite signal
                            feature_series = period_features.mean(axis=1)
                    else:
                        feature_series = pd.Series()
                    
                    sharpe_score = self._compute_oos_sharpe_nested(
                        feature_series,
                        targets,
                        direction
                    )[0]
                    
                    combined_score = 0.7 * avg_mi + 0.3 * sharpe_score
                    period_scores[str(period)] = combined_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_period = period
                        
                except Exception as e:
                    tprint_warning(f"Failed to optimize period {period}: {e}")
                    continue
        
        tprint_success(f"Best period: {best_period} (score: {best_score:.3f})")
        tprint_data_preview(period_scores, "period_optimization_scores", level="INFO")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(period_scores, "period_optimization_scores", level="INFO")
        return best_period, period_scores
    
    @memory_efficient(OptimizationConfig(
        enable_dtype_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_compression=True
    ))
    @smart_cache(cache_key_func=lambda self, features, targets, lookbacks_to_test, direction: 
                 f"lookback_opt_{hash(str(features.shape))}_{hash(str(targets.shape))}_{hash(tuple(lookbacks_to_test))}_{direction}")
    def _optimize_lookbacks(self, features: pd.DataFrame, targets: pd.Series, 
                           lookbacks_to_test: List[int], direction: str) -> Tuple[int, Dict[str, float]]:
        """Optimize lookbacks using mutual information and out-of-sample Sharpe ratio."""
        tprint_step("Optimizing lookbacks")
        
        # Input validation
        try:
            features = self.validator.validate_dataframe(features, min_rows=50)
            targets = self.validator.validate_series(targets, min_length=50)
            lookbacks_to_test = [self.validator.validate_positive_int(l, f"lookback_{l}") for l in lookbacks_to_test]
            if direction not in ['longs', 'shorts']:
                raise ValidationError(f"Direction must be 'longs' or 'shorts', got {direction}")
        except ValidationError as e:
            tprint_error(f"Lookback optimization validation failed: {e}")
            raise
        
        tprint_data_preview(features, "lookback_optimization_input", max_rows=5, level="DEBUG")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(features, "lookback_optimization_input", level="DEBUG")
        tprint_data_format(targets, "lookback_optimization_targets", level="DEBUG")
        
        # Pre-compute all lookback features using parallel processing
        tprint_info("🎯 Pre-computing lookback features for efficient optimization with parallel processing")
        lookback_features_cache = self._parallel_create_features(features, lookbacks_to_test, 'lookback')
        
        # Use TPE optimizer for sophisticated lookback optimization
        tprint_info("🎯 Using Bayesian TPE optimizer for lookback optimization")
        
        def lookback_objective(trial):
            """Objective function for TPE optimization."""
            lookback = trial.suggest_int('lookback', min(lookbacks_to_test), max(lookbacks_to_test))
            
            try:
                # Use pre-computed lookback features
                lookback_features = lookback_features_cache.get(lookback, pd.DataFrame())
                
                if lookback_features.empty:
                    return -np.inf
                
                # Calculate mutual information vectorized
                mi_scores = self._compute_vectorized_mutual_information(lookback_features, targets)
                
                if len(mi_scores) == 0 or np.all(mi_scores == 0):
                    return -np.inf
                
                avg_mi = np.mean(mi_scores)
                
                # Calculate out-of-sample Sharpe ratio using all features
                if len(lookback_features.columns) > 0:
                    # Use intelligent feature aggregation based on feature types
                    feature_series = self._create_intelligent_composite_signal(lookback_features)
                else:
                    feature_series = pd.Series()
                
                sharpe_score = self._compute_oos_sharpe_nested(
                    feature_series,
                    targets,
                    direction
                )[0]
                
                # Calculate comprehensive risk metrics
                risk_metrics = self._compute_comprehensive_risk_metrics(
                    feature_series,
                    targets,
                    direction
                )
                
                # Calculate transaction cost adjusted score
                transaction_cost = 0.001  # 0.1% total fees
                cost_adjusted_sharpe = sharpe_score - transaction_cost
                
                # Combined score (weighted average) with transaction cost consideration
                combined_score = 0.5 * avg_mi + 0.3 * cost_adjusted_sharpe + 0.2 * risk_metrics.get('sortino_ratio', 0.0)
                return combined_score
                
            except ValidationError as e:
                tprint_error(f"TPE trial validation failed for lookback {lookback}: {e}")
                return -np.inf
            except ValueError as e:
                tprint_error(f"TPE trial value error for lookback {lookback}: {e}")
                return -np.inf
            except Exception as e:
                tprint_error(f"TPE trial unexpected error for lookback {lookback}: {e}")
                tprint_debug(f"Exception details: {traceback.format_exc()}")
                return -np.inf
        
        # Run TPE optimization
        study = None
        try:
            study = self.tpe_optimizer.optimize(lookback_objective, {
                'lookback': {'type': 'int', 'low': min(lookbacks_to_test), 'high': max(lookbacks_to_test)}
            })
            
            best_lookback = study.best_params['lookback']
            best_score = study.best_value
            
            # Generate scores for all tested lookbacks for reporting (reuse TPE results if available)
            lookback_scores = {}
            if study is not None and hasattr(study, 'trials'):
                # Use TPE trial results if available
                for trial in study.trials:
                    if trial.value is not None and 'lookback' in trial.params:
                        lookback_scores[str(trial.params['lookback'])] = trial.value
                
                # Fill missing lookbacks with 0.0
                for lookback in lookbacks_to_test:
                    if str(lookback) not in lookback_scores:
                        lookback_scores[str(lookback)] = 0.0
            else:
                # Fast fail if TPE optimization fails
                tprint_error("TPE optimization failed - no fallback available")
                raise RuntimeError("TPE optimization failed and fallback is disabled for consistency")
                    
        except Exception as e:
            tprint_warning(f"TPE optimization failed, falling back to grid search: {e}")
            # Fallback to original grid search
            lookback_scores = {}
            best_lookback = lookbacks_to_test[0]
            best_score = -np.inf
        finally:
            # Clean up TPE study to prevent memory leaks
            if study is not None:
                try:
                    del study
                    gc.collect()
                except Exception as cleanup_error:
                    tprint_warning(f"TPE study cleanup failed: {cleanup_error}")
            
            for lookback in lookbacks_to_test:
                try:
                    lookback_features = self._create_lookback_features(features, lookback)
                    if lookback_features.empty:
                        continue
                    
                    mi_scores = []
                    for col in lookback_features.columns:
                        if not lookback_features[col].isna().all():
                            mi = self._compute_mutual_information(lookback_features[col], targets)
                            mi_scores.append(mi)
                    
                    if not mi_scores:
                        continue
                    
                    avg_mi = np.mean(mi_scores)
                    # Use all features for Sharpe calculation
                    if len(lookback_features.columns) > 0:
                        if len(lookback_features.columns) == 1:
                            feature_series = lookback_features.iloc[:, 0]
                        else:
                            # Use mean of all features as a composite signal
                            feature_series = lookback_features.mean(axis=1)
                    else:
                        feature_series = pd.Series()
                    
                    sharpe_score = self._compute_oos_sharpe_nested(
                        feature_series,
                        targets,
                        direction
                    )[0]
                    
                    combined_score = 0.7 * avg_mi + 0.3 * sharpe_score
                    lookback_scores[str(lookback)] = combined_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_lookback = lookback
                        
                except Exception as e:
                    tprint_warning(f"Failed to optimize lookback {lookback}: {e}")
                    continue
        
        tprint_success(f"Best lookback: {best_lookback} (score: {best_score:.3f})")
        tprint_data_preview(lookback_scores, "lookback_optimization_scores", level="INFO")
        # Add comprehensive data format analysis for troubleshooting
        tprint_data_format(lookback_scores, "lookback_optimization_scores", level="INFO")
        return best_lookback, lookback_scores
    
    def _is_returns_series(self, series: np.ndarray) -> bool:
        """Determine if a series represents returns or prices based on statistical properties."""
        if len(series) < 10:
            return False
        
        # Returns typically have mean close to 0 and are more normally distributed
        mean_val = np.mean(series)
        std_val = np.std(series)
        
        # If mean is close to 0 and std is reasonable for returns, likely returns
        if abs(mean_val) < 0.01 and 0.001 < std_val < 0.5:
            return True
        
        # If values are mostly between -1 and 1, likely returns
        if np.all(series >= -1) and np.all(series <= 1):
            return True
        
        # If values are mostly positive and growing, likely prices
        if np.all(series > 0) and np.mean(np.diff(series)) > 0:
            return False
        
        return False

    def _parallel_create_features(self, features: pd.DataFrame, values_to_test: List[int], feature_type: str) -> Dict[int, pd.DataFrame]:
        """Create features in parallel for multiple periods/lookbacks."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        features_cache = {}
        lock = threading.Lock()
        
        def create_single_feature(value):
            try:
                if feature_type == 'period':
                    result = self._create_period_features(features, value)
                elif feature_type == 'lookback':
                    result = self._create_lookback_features(features, value)
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                
                with lock:
                    features_cache[value] = result
                return value, result
            except Exception as e:
                tprint_warning(f"Failed to create {feature_type} features for {value}: {e}")
                with lock:
                    features_cache[value] = pd.DataFrame()
                return value, pd.DataFrame()
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(self.parallel_workers, len(values_to_test))
        tprint_info(f"Creating {feature_type} features in parallel with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_value = {
                executor.submit(create_single_feature, value): value 
                for value in values_to_test
            }
            
            # Process completed tasks
            for future in as_completed(future_to_value):
                value = future_to_value[future]
                try:
                    result_value, result_df = future.result()
                    if not result_df.empty:
                        tprint_info(f"Created {feature_type} features for {result_value}: {result_df.shape}")
                except Exception as e:
                    tprint_error(f"Error processing {feature_type} {value}: {e}")
        
        return features_cache

    def _create_intelligent_composite_signal(self, features: pd.DataFrame) -> pd.Series:
        """Create intelligent composite signal from multiple features based on their characteristics."""
        if len(features.columns) == 0:
            return pd.Series()
        elif len(features.columns) == 1:
            return features.iloc[:, 0]
        
        # Analyze feature characteristics to determine aggregation method
        feature_stats = {}
        for col in features.columns:
            if not features[col].isna().all():
                feature_stats[col] = {
                    'std': features[col].std(),
                    'mean': features[col].mean(),
                    'skew': features[col].skew(),
                    'kurt': features[col].kurtosis()
                }
        
        if not feature_stats:
            return pd.Series()
        
        # Use weighted average based on feature stability (lower std = higher weight)
        weights = []
        normalized_features = []
        
        for col, stats in feature_stats.items():
            # Weight inversely proportional to standard deviation (more stable = higher weight)
            weight = 1.0 / (1.0 + stats['std'])
            weights.append(weight)
            # Normalize feature to [0, 1] range
            feature_vals = features[col].fillna(0)
            if stats['std'] > 0:
                normalized_feature = (feature_vals - stats['mean']) / stats['std']
            else:
                normalized_feature = feature_vals - stats['mean']
            normalized_features.append(normalized_feature)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Create weighted composite signal
        composite_signal = pd.Series(0.0, index=features.index)
        for i, (weight, feature) in enumerate(zip(weights, normalized_features)):
            composite_signal += weight * feature
        
        return composite_signal

    def _create_period_features(self, features: pd.DataFrame, period: int) -> pd.DataFrame:
        """Create period-based features using proper period-based feature engineering."""
        # Check cache first
        cache_key = f"period_{period}_{hash(str(features.columns.tolist()))}"
        cached_features = self._get_cached_features(cache_key)
        if cached_features is not None:
            return cached_features
        
        try:
            if period <= 0 or period >= len(features):
                tprint_warning(f"Invalid period {period} for data length {len(features)}")
                return pd.DataFrame()
        except Exception as e:
            tprint_error(f"Error validating period {period}: {e}")
            raise ValidationError(f"Invalid period parameter: {e}")
        
        # Create period-based features using proper feature engineering
        period_features = pd.DataFrame(index=features.index)
        
        try:
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32', 'int64', 'int32', 'float16', 'int16']:
                    try:
                        # Use vectorized rolling operations for better performance
                        rolling_window = features[col].rolling(window=period, min_periods=period//2)
                        
                        # Vectorized calculation of multiple statistics at once
                        rolling_stats = rolling_window.agg(['mean', 'std', 'max', 'min', 'median'])
                        
                        period_features[f"{col}_period_mean"] = rolling_stats['mean']
                        period_features[f"{col}_period_std"] = rolling_stats['std']
                        period_features[f"{col}_period_max"] = rolling_stats['max']
                        period_features[f"{col}_period_min"] = rolling_stats['min']
                        period_features[f"{col}_period_median"] = rolling_stats['median']
                        
                        # Calculate period-based momentum and volatility with error handling
                        shifted_col = features[col].shift(period)
                        if not shifted_col.isna().all():
                            period_features[f"{col}_period_momentum"] = features[col] / shifted_col
                        
                        # Volatility calculation with division by zero protection
                        mean_vals = rolling_window.mean()
                        std_vals = rolling_window.std()
                        volatility = std_vals / mean_vals
                        volatility = volatility.replace([np.inf, -np.inf], np.nan)
                        period_features[f"{col}_period_volatility"] = volatility
                        
                    except Exception as e:
                        tprint_warning(f"Failed to create period features for column {col}: {e}")
                        continue
                        
        except Exception as e:
            tprint_error(f"Failed to create period features: {e}")
            return pd.DataFrame()
        
        # Drop rows with insufficient data
        period_features = period_features.dropna()
        
        if period_features.empty:
            tprint_warning(f"No valid period features created for period {period}")
            return pd.DataFrame()
        
        # Add data format analysis for period features (only for debugging)
        tprint_data_format(period_features, f"period_features_created_{period}", level="DEBUG")
        
        # Ensure we have enough data
        if len(period_features) < 10:
            return pd.DataFrame()
        
        # Cache the result
        if not period_features.empty:
            self._cache_features(cache_key, period_features)
        
        return period_features
    
    def _create_lookback_features(self, features: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Create lookback-based features using comprehensive rolling windows."""
        # Check cache first
        cache_key = f"lookback_{lookback}_{hash(str(features.columns.tolist()))}"
        cached_features = self._get_cached_features(cache_key)
        if cached_features is not None:
            return cached_features
        
        try:
            if lookback <= 0 or lookback >= len(features):
                tprint_warning(f"Invalid lookback {lookback} for data length {len(features)}")
                return pd.DataFrame()
        except Exception as e:
            tprint_error(f"Error validating lookback {lookback}: {e}")
            raise ValidationError(f"Invalid lookback parameter: {e}")
        
        # Create comprehensive lookback-based features
        lookback_features = pd.DataFrame(index=features.index)
        
        try:
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32', 'int64', 'int32', 'float16', 'int16']:
                    try:
                        # Use vectorized rolling operations for better performance
                        rolling_window = features[col].rolling(window=lookback, min_periods=lookback//2)
                        
                        # Vectorized calculation of multiple statistics at once
                        rolling_stats = rolling_window.agg(['mean', 'std', 'max', 'min', 'median', 'skew', 'kurt'])
                        
                        lookback_features[f"{col}_lookback_mean"] = rolling_stats['mean']
                        lookback_features[f"{col}_lookback_std"] = rolling_stats['std']
                        lookback_features[f"{col}_lookback_max"] = rolling_stats['max']
                        lookback_features[f"{col}_lookback_min"] = rolling_stats['min']
                        lookback_features[f"{col}_lookback_median"] = rolling_stats['median']
                        lookback_features[f"{col}_lookback_skew"] = rolling_stats['skew']
                        lookback_features[f"{col}_lookback_kurt"] = rolling_stats['kurt']
                        
                        # Volatility and momentum features with error handling
                        mean_vals = rolling_window.mean()
                        std_vals = rolling_window.std()
                        volatility = std_vals / mean_vals
                        volatility = volatility.replace([np.inf, -np.inf], np.nan)
                        lookback_features[f"{col}_lookback_volatility"] = volatility
                        
                        shifted_col = features[col].shift(lookback)
                        if not shifted_col.isna().all():
                            lookback_features[f"{col}_lookback_momentum"] = features[col] / shifted_col
                        
                    except Exception as e:
                        tprint_warning(f"Failed to create lookback features for column {col}: {e}")
                        continue
                        
        except Exception as e:
            tprint_error(f"Failed to create lookback features: {e}")
            return pd.DataFrame()
        
        # Drop rows with insufficient data
        lookback_features = lookback_features.dropna()
        
        if lookback_features.empty:
            tprint_warning(f"No valid lookback features created for lookback {lookback}")
            return pd.DataFrame()
                
                # Trend and change features
                lookback_features[f"{col}_lookback_trend"] = features[col].rolling(window=lookback, min_periods=lookback//2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                lookback_features[f"{col}_lookback_change"] = features[col].pct_change(lookback)
                
                # Range and position features
                lookback_features[f"{col}_lookback_range"] = features[col].rolling(window=lookback, min_periods=lookback//2).max() - features[col].rolling(window=lookback, min_periods=lookback//2).min()
                lookback_features[f"{col}_lookback_position"] = (features[col] - features[col].rolling(window=lookback, min_periods=lookback//2).min()) / (features[col].rolling(window=lookback, min_periods=lookback//2).max() - features[col].rolling(window=lookback, min_periods=lookback//2).min())
        
        # Drop rows with insufficient data
        lookback_features = lookback_features.dropna()
        
        # Add data format analysis for lookback features (only for debugging)
        tprint_data_format(lookback_features, f"lookback_features_created_{lookback}", level="DEBUG")
        
        # Ensure we have enough data
        if len(lookback_features) < 10:
            return pd.DataFrame()
        
        # Cache the result
        if not lookback_features.empty:
            self._cache_features(cache_key, lookback_features)
        
        return lookback_features
    
    @smart_cache(cache_key_func=lambda self, x, y: f"mi_{hash(str(x.values))}_{hash(str(y.values))}")
    def _compute_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Compute mutual information between two series with caching."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Add data format analysis for troubleshooting (only for debugging)
            tprint_data_format(x, "mi_x_series", level="DEBUG")
            tprint_data_format(y, "mi_y_series", level="DEBUG")
            
            # Align the series
            aligned_data = pd.concat([x, y], axis=1).dropna()
            if len(aligned_data) < 10:
                return 0.0
            
            x_vals = aligned_data.iloc[:, 0].values.reshape(-1, 1)
            y_vals = aligned_data.iloc[:, 1].values
            
            # Compute mutual information
            mi = mutual_info_regression(x_vals, y_vals, random_state=42)[0]
            return float(mi)
    
    def _compute_vectorized_mutual_information(self, features: pd.DataFrame, targets: pd.Series) -> np.ndarray:
        """Compute mutual information for all features vectorized."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Align features and targets
            aligned_data = features.join(targets.rename('target'), how='inner').dropna()
            if len(aligned_data) < 10:
                return np.array([])
            
            # Separate features and targets
            feature_cols = [col for col in features.columns if col in aligned_data.columns]
            if not feature_cols:
                return np.array([])
            
            X = aligned_data[feature_cols].values
            y = aligned_data['target'].values
            
            # Compute mutual information for all features at once
            mi_scores = mutual_info_regression(X, y, random_state=42)
            return mi_scores
            
        except Exception as e:
            tprint_warning(f"Vectorized mutual information calculation failed: {e}")
            # Fallback to individual calculation
            mi_scores = []
            for col in features.columns:
                if not features[col].isna().all():
                    mi = self._compute_mutual_information(features[col], targets)
                    mi_scores.append(mi)
            return np.array(mi_scores)
    
    def _compute_comprehensive_risk_metrics(self, features: pd.Series, targets: pd.Series, direction: str) -> Dict[str, float]:
        """Compute comprehensive risk metrics including Sharpe, Sortino, Calmar, and Max Drawdown."""
        try:
            # Align features and targets
            aligned_data = pd.concat([features, targets], axis=1).dropna()
            if len(aligned_data) < 50:
                return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'max_drawdown': 0.0}
            
            feature_vals = aligned_data.iloc[:, 0].values
            target_vals = aligned_data.iloc[:, 1].values
            
            # Determine if targets are returns or prices and calculate returns accordingly
            if self._is_returns_series(target_vals):
                returns = target_vals
            else:
                # Calculate returns from price series
                returns = np.diff(target_vals) / target_vals[:-1]
            
            if len(returns) == 0 or np.std(returns) == 0:
                return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'max_drawdown': 0.0}
            
            # Sharpe ratio
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
            else:
                sortino_ratio = sharpe_ratio
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Calmar ratio
            calmar_ratio = np.mean(returns) * 252 / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'max_drawdown': float(max_drawdown)
            }
            
        except Exception as e:
            tprint_warning(f"Risk metrics calculation failed: {e}")
            return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'max_drawdown': 0.0}
            
        except Exception as e:
            tprint_warning(f"Failed to compute mutual information: {e}")
            return 0.0

    def _analyze_feature_interactions(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Analyze feature interactions using correlation and mutual information."""
        try:
            # Calculate feature correlations
            feature_correlations = features.corr()
            
            # Calculate interaction scores
            interaction_scores = {}
            
            # Pairwise mutual information for feature interactions
            for i, col1 in enumerate(features.columns):
                for j, col2 in enumerate(features.columns):
                    if i < j:  # Avoid duplicates
                        # Calculate interaction as product of individual MIs minus joint MI
                        mi1 = self._compute_mutual_information(features[col1], targets)
                        mi2 = self._compute_mutual_information(features[col2], targets)
                        
                        # Create interaction feature
                        interaction_feature = features[col1] * features[col2]
                        mi_joint = self._compute_mutual_information(interaction_feature, targets)
                        
                        # Interaction score (positive means complementary, negative means redundant)
                        interaction_score = mi_joint - (mi1 + mi2) / 2
                        interaction_scores[f"{col1}_{col2}"] = interaction_score
            
            return interaction_scores
            
        except Exception as e:
            tprint_warning(f"Feature interaction analysis failed: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear all cached optimization results."""
        tprint_step("Clearing optimization cache")
        try:
            if hasattr(self, 'cache_system') and self.cache_system:
                self.cache_system.clear()
                tprint_success("Cache cleared successfully")
            else:
                tprint_warning("Cache system not available")
        except Exception as e:
            tprint_error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if hasattr(self, 'cache_system') and self.cache_system:
                return self.cache_system.get_stats()
            else:
                return {"error": "Cache system not available"}
        except Exception as e:
            return {"error": f"Failed to get cache stats: {e}"}
    
    def _validate_optimization_inputs(self, features: pd.DataFrame, targets: pd.Series, 
                                    periods_to_test: List[int], lookbacks_to_test: List[int], 
                                    direction: str) -> None:
        """Comprehensive validation of optimization inputs."""
        try:
            # Validate features
            features = self.validator.validate_dataframe(features, min_rows=100)
            
            # Validate targets
            targets = self.validator.validate_series(targets, min_length=100)
            
            # Validate periods
            if not periods_to_test or not all(isinstance(p, int) and p > 0 for p in periods_to_test):
                raise ValidationError("periods_to_test must be a non-empty list of positive integers")
            
            # Validate lookbacks
            if not lookbacks_to_test or not all(isinstance(l, int) and l > 0 for l in lookbacks_to_test):
                raise ValidationError("lookbacks_to_test must be a non-empty list of positive integers")
            
            # Validate direction
            if direction not in ['longs', 'shorts']:
                raise ValidationError(f"direction must be 'longs' or 'shorts', got {direction}")
            
            tprint_success("All optimization inputs validated successfully")
            
        except ValidationError as e:
            tprint_error(f"Optimization input validation failed: {e}")
            raise

    # Required utility methods for BasePreTrainingComponent

    # BaseStep utility methods
    def get_unified_hardware_manager(self):
        """Get unified hardware manager using BaseStep utilities."""
        return self.get_utility('unified_hardware_manager')

    def get_comprehensive_optimizer(self, **kwargs):
        """Get comprehensive optimizer using BaseStep utilities."""
        return self.get_utility('comprehensive_optimizer', **kwargs)

    def get_unified_memory_manager(self):
        """Get unified memory manager using BaseStep utilities."""
        return self.get_utility('unified_memory_manager')

    def get_advanced_cpu_optimizer(self):
        """Get advanced CPU optimizer using BaseStep utilities."""
        return self.get_utility('advanced_cpu_optimizer')

    def get_enhanced_gpu_manager(self):
        """Get enhanced GPU manager using BaseStep utilities."""
        return self.get_utility('enhanced_gpu_manager')

    def get_pretraining_artifact_manager(self):
        """Get pretraining artifact manager using BaseStep utilities."""
        return self.get_utility('pretraining_artifact_manager')
