"""
Step05 Intelligent Memory Management Module

This module provides intelligent memory management capabilities for Step05 labeling
operations, including memory optimization, garbage collection, and memory monitoring
with comprehensive logging.
"""

import pandas as pd
import numpy as np
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import weakref

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates
from src.utils.common_operations import safe_mean, safe_std, safe_float, safe_int, validate_dataframe_schema, validate_data_quality, safe_copy, safe_deepcopy, get_current_datetime, format_datetime, create_empty_dataframe, safe_fillna, safe_rolling, safe_append, safe_extend, safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join, get_logger, setup_basic_logging, safe_exception_handler, timed_operation, format_bytes, chunked_iterable, parallel_map, safe_log_metric, safe_log_params, safe_log_artifact
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive, validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change, validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.core.errors import AppError, ValidationError, DataIntegrityError, BusinessRuleError, NotFoundError, ConflictError, RateLimitError, TimeoutError, ServiceUnavailableError, ErrorCode

logger = system_logger.getChild('Step05MemoryManager')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    memory_percent: float = 0.0
    process_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryOptimizationResult:
    """Result of memory optimization operation."""
    original_memory_mb: float
    optimized_memory_mb: float
    reduction_percent: float
    optimization_time: float
    optimizations_applied: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class Step05MemoryManager:
    """Intelligent memory manager with optimization and monitoring capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.memory_history: List[MemoryStats] = []
        self.optimization_history: List[MemoryOptimizationResult] = []
        self.peak_memory_mb = 0.0
        self.memory_thresholds = {
            'warning_mb': 1000.0,  # 1GB warning threshold
            'critical_mb': 2000.0,  # 2GB critical threshold
            'max_memory_mb': 4000.0  # 4GB max memory limit
        }
        self.optimization_strategies = {
            'dtype_optimization': True,
            'categorical_optimization': True,
            'sparse_optimization': True,
            'chunk_processing': True,
            'garbage_collection': True
        }
        
        self._load_memory_config()
        self._initialize_memory_monitoring()
        
        self.logger.info("🚀 Initializing Step05 Intelligent Memory Manager")
        self.logger.info(f"💾 Memory thresholds: Warning={self.memory_thresholds['warning_mb']}MB, Critical={self.memory_thresholds['critical_mb']}MB")
        self.logger.info(f"🔧 Optimization strategies: {list(self.optimization_strategies.keys())}")
    
    def _load_memory_config(self):
        """Load memory configuration from config."""
        if 'memory' in self.config:
            memory_config = self.config['memory']
            
            # Load thresholds
            if 'thresholds' in memory_config:
                thresholds = memory_config['thresholds']
                self.memory_thresholds.update({
                    'warning_mb': thresholds.get('warning_mb', 1000.0),
                    'critical_mb': thresholds.get('critical_mb', 2000.0),
                    'max_memory_mb': thresholds.get('max_memory_mb', 4000.0)
                })
            
            # Load optimization strategies
            if 'optimization_strategies' in memory_config:
                strategies = memory_config['optimization_strategies']
                self.optimization_strategies.update({
                    'dtype_optimization': strategies.get('dtype_optimization', True),
                    'categorical_optimization': strategies.get('categorical_optimization', True),
                    'sparse_optimization': strategies.get('sparse_optimization', True),
                    'chunk_processing': strategies.get('chunk_processing', True),
                    'garbage_collection': strategies.get('garbage_collection', True)
                })
            
            self.logger.info("✅ Memory configuration loaded")
    
    def _initialize_memory_monitoring(self):
        """Initialize memory monitoring."""
        try:
            # Get initial memory stats
            initial_stats = self.get_memory_stats()
            self.memory_history.append(initial_stats)
            self.peak_memory_mb = initial_stats.process_memory_mb
            
            self.logger.info(f"💾 Initial memory usage: {initial_stats.process_memory_mb:.1f} MB")
            self.logger.info(f"🖥️ System memory: {initial_stats.used_memory_gb:.1f}GB / {initial_stats.total_memory_gb:.1f}GB ({initial_stats.memory_percent:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"❌ Memory monitoring initialization failed: {e}")
    
    @traced(span_name='get_memory_stats')
    @validates()
    @handles_errors()
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024**3)
            available_memory_gb = memory_info.available / (1024**3)
            used_memory_gb = memory_info.used / (1024**3)
            memory_percent = memory_info.percent
            
            # Process memory
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)
            
            # Update peak memory
            if process_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = process_memory_mb
            
            stats = MemoryStats(
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                used_memory_gb=used_memory_gb,
                memory_percent=memory_percent,
                process_memory_mb=process_memory_mb,
                peak_memory_mb=self.peak_memory_mb
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory stats: {e}")
            return MemoryStats()
    
    @traced(span_name='monitor_memory_usage')
    @validates()
    @handles_errors()
    def monitor_memory_usage(self, operation_name: str = "operation") -> MemoryStats:
        """Monitor memory usage for an operation."""
        try:
            stats = self.get_memory_stats()
            self.memory_history.append(stats)
            
            # Log memory usage
            self.logger.info(f"💾 Memory usage for {operation_name}: {stats.process_memory_mb:.1f} MB")
            
            # Check thresholds
            if stats.process_memory_mb > self.memory_thresholds['critical_mb']:
                self.logger.error(f"🚨 CRITICAL: Memory usage exceeds critical threshold: {stats.process_memory_mb:.1f} MB > {self.memory_thresholds['critical_mb']} MB")
                self._trigger_emergency_cleanup()
            elif stats.process_memory_mb > self.memory_thresholds['warning_mb']:
                self.logger.warning(f"⚠️ WARNING: Memory usage exceeds warning threshold: {stats.process_memory_mb:.1f} MB > {self.memory_thresholds['warning_mb']} MB")
                self._trigger_memory_optimization()
            
            # Check system memory
            if stats.memory_percent > 90:
                self.logger.error(f"🚨 CRITICAL: System memory usage critical: {stats.memory_percent:.1f}%")
            elif stats.memory_percent > 80:
                self.logger.warning(f"⚠️ WARNING: System memory usage high: {stats.memory_percent:.1f}%")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Memory monitoring failed: {e}")
            return MemoryStats()
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        try:
            self.logger.warning("🚨 Triggering emergency memory cleanup...")
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"🗑️ Emergency garbage collection: {collected} objects collected")
            
            # Clear memory history (keep only last 10 entries)
            if len(self.memory_history) > 10:
                self.memory_history = self.memory_history[-10:]
                self.logger.info("🗑️ Cleared old memory history")
            
            # Clear optimization history (keep only last 5 entries)
            if len(self.optimization_history) > 5:
                self.optimization_history = self.optimization_history[-5:]
                self.logger.info("🗑️ Cleared old optimization history")
            
            # Get memory stats after cleanup
            stats = self.get_memory_stats()
            self.logger.info(f"💾 Memory after emergency cleanup: {stats.process_memory_mb:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency cleanup failed: {e}")
    
    def _trigger_memory_optimization(self):
        """Trigger memory optimization."""
        try:
            self.logger.info("🔧 Triggering memory optimization...")
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"🗑️ Garbage collection: {collected} objects collected")
            
            # Get memory stats after optimization
            stats = self.get_memory_stats()
            self.logger.info(f"💾 Memory after optimization: {stats.process_memory_mb:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"❌ Memory optimization failed: {e}")
    
    @traced(span_name='optimize_dataframe_memory')
    @validates()
    @handles_errors()
    def optimize_dataframe_memory(self, data: pd.DataFrame, 
                                 operation_name: str = "dataframe_optimization") -> MemoryOptimizationResult:
        """
        Optimize DataFrame memory usage with comprehensive logging.
        
        Args:
            data: DataFrame to optimize
            operation_name: Name of the operation for logging
            
        Returns:
            MemoryOptimizationResult with optimization details
        """
        start_time = time.time()
        self.logger.info(f"🔧 Starting DataFrame memory optimization: {operation_name}")
        
        try:
            # Get initial memory usage
            initial_stats = self.monitor_memory_usage(f"{operation_name}_initial")
            original_memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
            
            self.logger.info(f"📊 Original DataFrame memory usage: {original_memory_mb:.1f} MB")
            self.logger.info(f"📊 DataFrame shape: {data.shape}")
            self.logger.info(f"📋 DataFrame columns: {list(data.columns)}")
            
            optimizations_applied = []
            warnings = []
            details = {}
            
            # 1. Numeric column optimization
            if self.optimization_strategies['dtype_optimization']:
                self.logger.info("🔧 Optimizing numeric column data types...")
                
                numeric_optimizations = self._optimize_numeric_columns(data)
                optimizations_applied.extend(numeric_optimizations['applied'])
                warnings.extend(numeric_optimizations['warnings'])
                details['numeric_optimizations'] = numeric_optimizations['details']
                
                self.logger.info(f"✅ Numeric optimizations: {len(numeric_optimizations['applied'])} applied")
            
            # 2. Categorical column optimization
            if self.optimization_strategies['categorical_optimization']:
                self.logger.info("🔧 Optimizing categorical columns...")
                
                categorical_optimizations = self._optimize_categorical_columns(data)
                optimizations_applied.extend(categorical_optimizations['applied'])
                warnings.extend(categorical_optimizations['warnings'])
                details['categorical_optimizations'] = categorical_optimizations['details']
                
                self.logger.info(f"✅ Categorical optimizations: {len(categorical_optimizations['applied'])} applied")
            
            # 3. Sparse data optimization
            if self.optimization_strategies['sparse_optimization']:
                self.logger.info("🔧 Optimizing sparse data...")
                
                sparse_optimizations = self._optimize_sparse_data(data)
                optimizations_applied.extend(sparse_optimizations['applied'])
                warnings.extend(sparse_optimizations['warnings'])
                details['sparse_optimizations'] = sparse_optimizations['details']
                
                self.logger.info(f"✅ Sparse optimizations: {len(sparse_optimizations['applied'])} applied")
            
            # 4. Index optimization
            self.logger.info("🔧 Optimizing DataFrame index...")
            
            index_optimizations = self._optimize_dataframe_index(data)
            optimizations_applied.extend(index_optimizations['applied'])
            warnings.extend(index_optimizations['warnings'])
            details['index_optimizations'] = index_optimizations['details']
            
            self.logger.info(f"✅ Index optimizations: {len(index_optimizations['applied'])} applied")
            
            # Calculate final memory usage
            optimized_memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
            reduction_percent = (original_memory_mb - optimized_memory_mb) / original_memory_mb * 100
            
            optimization_time = time.time() - start_time
            
            # Get final memory stats
            final_stats = self.monitor_memory_usage(f"{operation_name}_final")
            
            result = MemoryOptimizationResult(
                original_memory_mb=original_memory_mb,
                optimized_memory_mb=optimized_memory_mb,
                reduction_percent=reduction_percent,
                optimization_time=optimization_time,
                optimizations_applied=optimizations_applied,
                warnings=warnings,
                details=details
            )
            
            # Store optimization result
            self.optimization_history.append(result)
            
            # Log detailed results
            self.logger.info(f"✅ DataFrame memory optimization completed in {optimization_time:.3f}s")
            self.logger.info(f"💾 Memory reduction: {reduction_percent:.1f}% ({original_memory_mb:.1f}MB → {optimized_memory_mb:.1f}MB)")
            self.logger.info(f"🔧 Optimizations applied: {len(optimizations_applied)}")
            self.logger.info(f"⚠️ Warnings: {len(warnings)}")
            
            if warnings:
                self.logger.warning("⚠️ Optimization warnings:")
                for warning in warnings:
                    self.logger.warning(f"   - {warning}")
            
            return result
            
        except Exception as e:
            optimization_time = time.time() - start_time
            self.logger.error(f"❌ DataFrame memory optimization failed after {optimization_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            return MemoryOptimizationResult(
                original_memory_mb=0.0, optimized_memory_mb=0.0, reduction_percent=0.0,
                optimization_time=optimization_time, optimizations_applied=[], warnings=[f"Optimization failed: {str(e)}"],
                details={'error': str(e)}
            )
    
    def _optimize_numeric_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize numeric column data types."""
        optimizations_applied = []
        warnings = []
        details = {}
        
        try:
            # Integer columns
            int_columns = data.select_dtypes(include=['int64']).columns
            for col in int_columns:
                if data[col].min() >= 0:  # Unsigned integers
                    if data[col].max() < 255:
                        data[col] = data[col].astype('uint8')
                        optimizations_applied.append(f"Converted {col} to uint8")
                    elif data[col].max() < 65535:
                        data[col] = data[col].astype('uint16')
                        optimizations_applied.append(f"Converted {col} to uint16")
                    elif data[col].max() < 4294967295:
                        data[col] = data[col].astype('uint32')
                        optimizations_applied.append(f"Converted {col} to uint32")
                else:  # Signed integers
                    if data[col].min() >= -128 and data[col].max() <= 127:
                        data[col] = data[col].astype('int8')
                        optimizations_applied.append(f"Converted {col} to int8")
                    elif data[col].min() >= -32768 and data[col].max() <= 32767:
                        data[col] = data[col].astype('int16')
                        optimizations_applied.append(f"Converted {col} to int16")
                    elif data[col].min() >= -2147483648 and data[col].max() <= 2147483647:
                        data[col] = data[col].astype('int32')
                        optimizations_applied.append(f"Converted {col} to int32")
            
            # Float columns
            float_columns = data.select_dtypes(include=['float64']).columns
            for col in float_columns:
                # Check if float32 is sufficient
                if data[col].min() >= np.finfo(np.float32).min and data[col].max() <= np.finfo(np.float32).max:
                    # Check precision loss
                    original_precision = data[col].astype('float32').astype('float64')
                    precision_loss = (data[col] - original_precision).abs().max()
                    
                    if precision_loss < 1e-6:  # Acceptable precision loss
                        data[col] = data[col].astype('float32')
                        optimizations_applied.append(f"Converted {col} to float32")
                    else:
                        warnings.append(f"Precision loss too high for {col}: {precision_loss:.2e}")
            
            details = {
                'int_columns_optimized': len([opt for opt in optimizations_applied if 'int' in opt]),
                'float_columns_optimized': len([opt for opt in optimizations_applied if 'float' in opt])
            }
            
        except Exception as e:
            warnings.append(f"Numeric optimization failed: {str(e)}")
        
        return {
            'applied': optimizations_applied,
            'warnings': warnings,
            'details': details
        }
    
    def _optimize_categorical_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize categorical columns."""
        optimizations_applied = []
        warnings = []
        details = {}
        
        try:
            # Object columns that could be categorical
            object_columns = data.select_dtypes(include=['object']).columns
            
            for col in object_columns:
                unique_ratio = data[col].nunique() / len(data)
                
                if unique_ratio < 0.5:  # Less than 50% unique values
                    # Check memory savings
                    original_memory = data[col].memory_usage(deep=True)
                    categorical_memory = data[col].astype('category').memory_usage(deep=True)
                    
                    if categorical_memory < original_memory:
                        data[col] = data[col].astype('category')
                        optimizations_applied.append(f"Converted {col} to category")
                    else:
                        warnings.append(f"No memory savings for categorical conversion of {col}")
            
            details = {
                'categorical_columns_created': len([opt for opt in optimizations_applied if 'category' in opt]),
                'object_columns_analyzed': len(object_columns)
            }
            
        except Exception as e:
            warnings.append(f"Categorical optimization failed: {str(e)}")
        
        return {
            'applied': optimizations_applied,
            'warnings': warnings,
            'details': details
        }
    
    def _optimize_sparse_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize sparse data."""
        optimizations_applied = []
        warnings = []
        details = {}
        
        try:
            # Check for sparse columns (high percentage of zeros or nulls)
            for col in data.columns:
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Check for zeros
                    zero_ratio = (data[col] == 0).sum() / len(data)
                    
                    if zero_ratio > 0.8:  # More than 80% zeros
                        # Check if sparse representation would save memory
                        original_memory = data[col].memory_usage(deep=True)
                        
                        try:
                            sparse_data = data[col].astype(pd.SparseDtype(data[col].dtype, 0))
                            sparse_memory = sparse_data.memory_usage(deep=True)
                            
                            if sparse_memory < original_memory:
                                data[col] = sparse_data
                                optimizations_applied.append(f"Converted {col} to sparse (zero_ratio: {zero_ratio:.1%})")
                        except Exception as e:
                            warnings.append(f"Sparse conversion failed for {col}: {str(e)}")
            
            details = {
                'sparse_columns_created': len([opt for opt in optimizations_applied if 'sparse' in opt])
            }
            
        except Exception as e:
            warnings.append(f"Sparse optimization failed: {str(e)}")
        
        return {
            'applied': optimizations_applied,
            'warnings': warnings,
            'details': details
        }
    
    def _optimize_dataframe_index(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize DataFrame index."""
        optimizations_applied = []
        warnings = []
        details = {}
        
        try:
            # Check if index can be optimized
            if hasattr(data.index, 'dtype'):
                index_dtype = data.index.dtype
                
                # Integer index optimization
                if 'int' in str(index_dtype):
                    if data.index.min() >= 0:  # Unsigned
                        if data.index.max() < 4294967295:
                            data.index = data.index.astype('uint32')
                            optimizations_applied.append("Optimized index to uint32")
                    else:  # Signed
                        if data.index.min() >= -2147483648 and data.index.max() <= 2147483647:
                            data.index = data.index.astype('int32')
                            optimizations_applied.append("Optimized index to int32")
                
                # Datetime index optimization
                elif 'datetime' in str(index_dtype):
                    # Check if we can use a more efficient datetime representation
                    if hasattr(data.index, 'freq') and data.index.freq is not None:
                        optimizations_applied.append("Datetime index has regular frequency")
            
            details = {
                'index_type': str(data.index.dtype),
                'index_optimizations': len(optimizations_applied)
            }
            
        except Exception as e:
            warnings.append(f"Index optimization failed: {str(e)}")
        
        return {
            'applied': optimizations_applied,
            'warnings': warnings,
            'details': details
        }
    
    @traced(span_name='process_with_memory_management')
    @validates()
    @handles_errors()
    def process_with_memory_management(self, data: pd.DataFrame, 
                                     processing_function: Callable,
                                     operation_name: str = "memory_managed_processing") -> Any:
        """
        Process data with intelligent memory management.
        
        Args:
            data: DataFrame to process
            processing_function: Function to process the data
            operation_name: Name of the operation for logging
            
        Returns:
            Processing result
        """
        start_time = time.time()
        self.logger.info(f"🔧 Starting memory-managed processing: {operation_name}")
        
        try:
            # Monitor initial memory
            initial_stats = self.monitor_memory_usage(f"{operation_name}_start")
            
            # Optimize input data if needed
            if initial_stats.process_memory_mb > self.memory_thresholds['warning_mb']:
                self.logger.info("🔧 Input data memory usage high, optimizing...")
                optimization_result = self.optimize_dataframe_memory(data, f"{operation_name}_input")
                self.logger.info(f"💾 Input optimization: {optimization_result.reduction_percent:.1f}% reduction")
            
            # Process data
            self.logger.info("⚙️ Processing data...")
            result = processing_function(data)
            
            # Monitor memory after processing
            processing_stats = self.monitor_memory_usage(f"{operation_name}_processing")
            
            # Optimize result if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                if processing_stats.process_memory_mb > self.memory_thresholds['warning_mb']:
                    self.logger.info("🔧 Result memory usage high, optimizing...")
                    optimization_result = self.optimize_dataframe_memory(result, f"{operation_name}_result")
                    self.logger.info(f"💾 Result optimization: {optimization_result.reduction_percent:.1f}% reduction")
            
            # Final memory check
            final_stats = self.monitor_memory_usage(f"{operation_name}_end")
            
            processing_time = time.time() - start_time
            
            # Log processing summary
            memory_delta = final_stats.process_memory_mb - initial_stats.process_memory_mb
            self.logger.info(f"✅ Memory-managed processing completed in {processing_time:.3f}s")
            self.logger.info(f"💾 Memory delta: {memory_delta:+.1f} MB")
            self.logger.info(f"📊 Peak memory: {self.peak_memory_mb:.1f} MB")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Memory-managed processing failed after {processing_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            return None
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        try:
            current_stats = self.get_memory_stats()
            
            return {
                'current_memory': {
                    'process_memory_mb': current_stats.process_memory_mb,
                    'system_memory_percent': current_stats.memory_percent,
                    'available_memory_gb': current_stats.available_memory_gb
                },
                'peak_memory_mb': self.peak_memory_mb,
                'memory_thresholds': self.memory_thresholds.copy(),
                'optimization_history': [
                    {
                        'reduction_percent': opt.reduction_percent,
                        'optimization_time': opt.optimization_time,
                        'optimizations_applied': len(opt.optimizations_applied)
                    }
                    for opt in self.optimization_history[-5:]  # Last 5 optimizations
                ],
                'memory_history_count': len(self.memory_history),
                'optimization_strategies': self.optimization_strategies.copy()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory summary: {e}")
            return {'error': str(e)}
    
    def cleanup_memory(self):
        """Perform comprehensive memory cleanup."""
        try:
            self.logger.info("🧹 Starting comprehensive memory cleanup...")
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"🗑️ Garbage collection: {collected} objects collected")
            
            # Clear history (keep recent entries)
            if len(self.memory_history) > 20:
                self.memory_history = self.memory_history[-20:]
                self.logger.info("🗑️ Cleared old memory history")
            
            if len(self.optimization_history) > 10:
                self.optimization_history = self.optimization_history[-10:]
                self.logger.info("🗑️ Cleared old optimization history")
            
            # Get final memory stats
            final_stats = self.get_memory_stats()
            self.logger.info(f"💾 Memory after cleanup: {final_stats.process_memory_mb:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"❌ Memory cleanup failed: {e}")