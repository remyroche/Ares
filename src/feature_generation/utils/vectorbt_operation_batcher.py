"""
VectorBT Operation Batcher

This module provides optimized batching for VectorBT operations to reduce overhead
and improve performance when processing multiple similar operations.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import os

# VectorBT imports
try:
    import vectorbt as vbt
    try:
        # Prefer explicit generic rolling functions when available
        from src.utils.vectorbt_compat import (
            rolling_mean, rolling_std, rolling_var, rolling_min,
            rolling_max, rolling_sum
        )
    except Exception:
        rolling_mean = rolling_std = rolling_var = rolling_min = None
        rolling_max = rolling_sum = None
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = rolling_std = rolling_var = rolling_min = None
    rolling_max = rolling_sum = None

logger = logging.getLogger(__name__)

@dataclass
class VectorBTOperation:
    """Represents a VectorBT operation to be batched."""
    name: str
    operation_func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0  # Higher priority operations are processed first
    memory_weight: float = 1.0  # Estimated memory usage weight

@dataclass
class VectorBTBatchConfig:
    """Configuration for VectorBT operation batching."""
    max_batch_size: int = 50
    max_memory_mb: float = 512.0
    enable_parallel_processing: bool = True
    max_workers: int = 4
    operation_timeout: float = 30.0
    enable_memory_optimization: bool = True
    chunk_size: int = 1000

class VectorBTOperationBatcher:
    """
    Batches VectorBT operations for improved performance and reduced overhead.
    
    This class groups similar VectorBT operations together and processes them
    in batches to minimize the overhead of repeated function calls and
    memory allocations.
    """
    
    def __init__(self, config: Optional[VectorBTBatchConfig] = None):
        """
        Initialize the VectorBT operation batcher.
        
        Args:
            config: Configuration for the batcher
        """
        self.config = config or VectorBTBatchConfig()
        self.logger = logger.getChild('VectorBTOperationBatcher')
        self.operations_queue: List[VectorBTOperation] = []
        self.results_cache: Dict[str, Any] = {}
        
        if not VECTORBT_AVAILABLE:
            self.logger.warning("VectorBT not available - batching will be disabled")
    
    def add_operation(self, 
                     name: str,
                     operation_func: Callable,
                     *args,
                     priority: int = 0,
                     memory_weight: float = 1.0,
                     **kwargs) -> None:
        """
        Add an operation to the batch queue.
        
        Args:
            name: Unique name for the operation
            operation_func: VectorBT function to execute
            *args: Arguments for the operation
            priority: Priority level (higher = processed first)
            memory_weight: Estimated memory usage weight
            **kwargs: Keyword arguments for the operation
        """
        # Allow queuing even if VectorBT is unavailable; operation_func may be a pandas fallback
            
        operation = VectorBTOperation(
            name=name,
            operation_func=operation_func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            memory_weight=memory_weight
        )
        
        self.operations_queue.append(operation)
        self.logger.debug(f"Added operation: {name}")
    
    def execute_batch(self) -> Dict[str, Any]:
        """
        Execute all queued operations in optimized batches.
        
        Returns:
            Dictionary mapping operation names to their results
        """
        if not self.operations_queue:
            return {}
        
        if not VECTORBT_AVAILABLE:
            self.logger.warning("VectorBT not available - skipping batch execution")
            return {}
        
        self.logger.info(f"Executing batch of {len(self.operations_queue)} operations")
        
        # Sort operations by priority (higher priority first)
        sorted_operations = sorted(self.operations_queue, key=lambda op: op.priority, reverse=True)
        
        # Group operations into batches
        batches = self._create_optimized_batches(sorted_operations)
        
        results = {}
        
        for batch_num, batch in enumerate(batches, 1):
            self.logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} operations)")
            
            try:
                batch_results = self._execute_batch_parallel(batch)
                results.update(batch_results)
            except Exception as e:
                self.logger.error(f"Error executing batch {batch_num}: {e}")
                # Continue with other batches
                continue
        
        # Clear the queue and results cache
        self.operations_queue.clear()
        self.results_cache.clear()
        
        # Force garbage collection after batch execution
        import gc
        collected = gc.collect()
        if collected > 0:
            self.logger.debug(f"🧹 Batch cleanup: {collected} objects collected")
        
        self.logger.info(f"Batch execution completed: {len(results)} results")
        return results
    
    def _create_optimized_batches(self, operations: List[VectorBTOperation]) -> List[List[VectorBTOperation]]:
        """
        Create optimized batches based on memory usage and operation similarity.
        
        Args:
            operations: List of sorted operations
            
        Returns:
            List of operation batches
        """
        batches = []
        current_batch = []
        current_memory_weight = 0.0
        
        for operation in operations:
            # Check if adding this operation would exceed limits
            if (len(current_batch) >= self.config.max_batch_size or 
                current_memory_weight + operation.memory_weight > self.config.max_memory_mb / 100):  # Rough MB estimate
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_memory_weight = 0.0
            
            current_batch.append(operation)
            current_memory_weight += operation.memory_weight
        
        # Add the last batch if it has operations
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _execute_batch_parallel(self, batch: List[VectorBTOperation]) -> Dict[str, Any]:
        """
        Execute a batch of operations in parallel.
        
        Args:
            batch: List of operations to execute
            
        Returns:
            Dictionary of operation results
        """
        results = {}
        
        if self.config.enable_parallel_processing and len(batch) > 1:
            # Execute in parallel
            with ThreadPoolExecutor(max_workers=self._resolve_max_workers()) as executor:
                future_to_operation = {
                    executor.submit(self._execute_single_operation, operation): operation
                    for operation in batch
                }
                
                for future in as_completed(future_to_operation, timeout=self.config.operation_timeout):
                    operation = future_to_operation[future]
                    try:
                        result = future.result(timeout=self.config.operation_timeout)
                        results[operation.name] = result
                    except Exception as e:
                        self.logger.error(f"Error executing operation {operation.name}: {e}")
                        results[operation.name] = None
        else:
            # Execute sequentially
            for operation in batch:
                try:
                    result = self._execute_single_operation(operation)
                    results[operation.name] = result
                except Exception as e:
                    self.logger.error(f"Error executing operation {operation.name}: {e}")
                    results[operation.name] = None
        
        return results

    def _execute_single_operation(self, operation: VectorBTOperation) -> Any:
        """
        Execute a single VectorBT operation.
        
        Args:
            operation: Operation to execute
            
        Returns:
            Result of the operation
        """
        start_time = time.time()
        
        try:
            # Prepare inputs: enforce float32 for numeric data without unnecessary copies
            def _cast_arg(arg):
                try:
                    if isinstance(arg, pd.DataFrame):
                        df = arg
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        if len(num_cols) > 0:
                            df[num_cols] = df[num_cols].astype(np.float32, copy=False)
                        return df
                    if isinstance(arg, pd.Series):
                        if pd.api.types.is_numeric_dtype(arg):
                            return arg.astype(np.float32, copy=False)
                        return arg
                except Exception:
                    return arg
                return arg

            new_args = tuple(_cast_arg(a) for a in operation.args)
            new_kwargs = {k: _cast_arg(v) for k, v in operation.kwargs.items()}

            # Execute the operation
            result = operation.operation_func(*new_args, **new_kwargs)
            
            execution_time = time.time() - start_time
            self.logger.debug(f"Operation {operation.name} completed in {execution_time:.3f}s")
            
            # Apply memory optimization if enabled
            if self.config.enable_memory_optimization:
                result = self._optimize_result_memory(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in operation {operation.name}: {e}")
            raise
    
    def _optimize_result_memory(self, result: Any) -> Any:
        """
        Optimize memory usage of operation results.
        
        Args:
            result: Operation result to optimize
            
        Returns:
            Memory-optimized result
        """
        if isinstance(result, pd.DataFrame):
            # Optimize DataFrame memory usage
            return self._optimize_dataframe_memory(result)
        elif isinstance(result, pd.Series):
            # Optimize Series memory usage
            return self._optimize_series_memory(result)
        else:
            return result
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        # Convert float64 to float32 if possible
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32, copy=False)
        
        # Convert int64 to int32 if possible
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32, copy=False)
        
        return df
    
    def _optimize_series_memory(self, series: pd.Series) -> pd.Series:
        """
        Optimize Series memory usage.
        
        Args:
            series: Series to optimize
            
        Returns:
            Memory-optimized Series
        """
        # Convert float64 to float32 if possible
        if series.dtype == 'float64':
            if series.min() >= np.finfo(np.float32).min and series.max() <= np.finfo(np.float32).max:
                series = series.astype(np.float32, copy=False)
        
        # Convert int64 to int32 if possible
        elif series.dtype == 'int64':
            if series.min() >= np.iinfo(np.int32).min and series.max() <= np.iinfo(np.int32).max:
                series = series.astype(np.int32, copy=False)

        return series

    def _resolve_max_workers(self) -> int:
        """Resolve a safe max_workers to avoid CPU oversubscription."""
        try:
            # Prefer a conservative cap based on available CPUs
            cpu_count = os.cpu_count() or 4
            cfg_workers = int(self.config.max_workers or cpu_count)
            return max(1, min(cfg_workers, cpu_count))
        except Exception:
            return max(1, self.config.max_workers or 4)

    def generate_features_batch_optimized(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate multiple features in batch with memory and performance optimization.
        
        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            Dictionary mapping feature names to their generated series
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("VectorBT not available - falling back to pandas operations")
            return {}
        
        if not feature_configs:
            return {}
        
        self.logger.info(f"🚀 Generating {len(feature_configs)} features in batch...")
        
        # Clear any existing operations
        self.operations_queue.clear()
        
        # Add operations for each feature configuration
        for config in feature_configs:
            feature_name = config.get('name')
            if not feature_name:
                continue
                
            # Determine operation type and add to batch
            feature_type = config.get('type', 'indicator')
            params = config.get('params', {})
            
            if feature_type == 'rolling':
                # Add rolling operations
                operation = params.get('operation', 'mean')
                window = params.get('window', 20)
                column = params.get('column', 'close')
                
                if operation == 'mean':
                    self.add_operation(
                        feature_name,
                        rolling_mean if rolling_mean else lambda x, **kw: x.rolling(**kw).mean(),
                        *(data[column] if column in data.columns else data.iloc[:, 0],),
                        window=window,
                        priority=1,
                        memory_weight=1.0
                    )
                elif operation == 'std':
                    self.add_operation(
                        feature_name,
                        rolling_std if rolling_std else lambda x, **kw: x.rolling(**kw).std(),
                        *(data[column] if column in data.columns else data.iloc[:, 0],),
                        window=window,
                        priority=1,
                        memory_weight=1.0
                    )
                # Add other rolling operations as needed
                
            elif feature_type == 'indicator':
                # Add technical indicator operations
                indicator_type = params.get('indicator', 'sma')
                
                if indicator_type == 'sma':
                    period = params.get('period', 20)
                    column = params.get('column', 'close')
                    self.add_operation(
                        feature_name,
                        rolling_mean if rolling_mean else lambda x, **kw: x.rolling(**kw).mean(),
                        *(data[column] if column in data.columns else data.iloc[:, 0],),
                        window=period,
                        priority=2,
                        memory_weight=1.0
                    )
                # Add other indicator types as needed
        
        # Execute the batch and return results
        results = self.execute_batch()
        
        # Convert results to proper format (Series instead of raw results)
        formatted_results = {}
        for feature_name, result in results.items():
            if result is not None:
                if isinstance(result, pd.Series):
                    formatted_results[feature_name] = result
                elif isinstance(result, pd.DataFrame):
                    # Take the first column if DataFrame
                    formatted_results[feature_name] = result.iloc[:, 0]
                else:
                    # Try to convert to Series
                    try:
                        formatted_results[feature_name] = pd.Series(result, index=data.index)
                    except Exception as e:
                        self.logger.warning(f"Could not convert result for {feature_name}: {e}")
        
        self.logger.info(f"✅ Generated {len(formatted_results)} features successfully")
        return formatted_results

    # ------------------------------
    # Wide-matrix rolling kernels
    # ------------------------------
    def rolling_dataframe(self, data: pd.DataFrame, window_sizes: List[int],
                          operation: str = "mean", min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Apply a rolling operation to all numeric columns of a DataFrame, coalesced per window.

        Args:
            data: Input DataFrame
            window_sizes: List of window sizes
            operation: One of {'mean','std','var','min','max','sum'}
            min_periods: Optional min_periods

        Returns:
            DataFrame with columns suffixed as '<col>_<operation>_<window>'
        """
        if data is None or data.empty:
            return pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else None)

        op = operation.lower()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame(index=data.index)

        # Ensure float32 for numeric columns, no copy when possible
        df = data.copy() if not set(numeric_cols).issubset(set(data.columns)) else data
        try:
            df[numeric_cols] = df[numeric_cols].astype(np.float32, copy=False)
        except Exception:
            pass

        results = []
        for w in sorted(set(int(w) for w in window_sizes)):
            try:
                if VECTORBT_AVAILABLE and (rolling_mean is not None):
                    # Use mapping to vectorbt functions if available
                    vmap = {
                        'mean': rolling_mean,
                        'std': rolling_std,
                        'var': rolling_var,
                        'min': rolling_min,
                        'max': rolling_max,
                        'sum': rolling_sum,
                    }
                    func = vmap.get(op)
                    if func is None:
                        raise ValueError(f"Unsupported operation: {operation}")
                    rolled = func(df[numeric_cols], window=w) if min_periods is None else func(df[numeric_cols], window=w, min_periods=min_periods)
                else:
                    # Pandas fallback across the entire frame
                    rolled = df[numeric_cols].rolling(window=w, min_periods=(min_periods or w))
                    if op == 'mean':
                        rolled = rolled.mean()
                    elif op == 'std':
                        rolled = rolled.std()
                    elif op == 'var':
                        rolled = rolled.var()
                    elif op == 'min':
                        rolled = rolled.min()
                    elif op == 'max':
                        rolled = rolled.max()
                    elif op == 'sum':
                        rolled = rolled.sum()
                    else:
                        raise ValueError(f"Unsupported operation: {operation}")

                # Rename columns with suffix
                rolled = rolled.rename(columns={c: f"{c}_{op}_{w}" for c in rolled.columns})
                results.append(rolled)
            except Exception as e:
                self.logger.error(f"Wide rolling failed for window {w} op {op}: {e}")
                continue

        if not results:
            return pd.DataFrame(index=df.index)

        out = pd.concat(results, axis=1)
        # Optimize memory for the output
        try:
            out = self._optimize_dataframe_memory(out)
        except Exception:
            pass
        return out

# Convenience functions for common VectorBT operations
def batch_rolling_operations(data: pd.DataFrame,
                           window_sizes: List[int],
                           operation_name: str = "mean",
                           batcher: Optional[VectorBTOperationBatcher] = None) -> Dict[str, Any]:
    """
    Batch multiple rolling operations on the same data.
    
    Args:
        data: Input data
        window_sizes: List of window sizes for rolling operations
        operation_name: Base name for operations
        batcher: Optional batcher instance
        
    Returns:
        Dictionary of rolling operation results
    """
    if batcher is None:
        batcher = VectorBTOperationBatcher()

    # Map operation name to function (VectorBT if available, else pandas fallback)
    def _pandas_rolling_func(op: str) -> Callable:
        if op == 'mean':
            return lambda x, *, window, **kw: x.rolling(window=window, **kw).mean()
        if op == 'std':
            return lambda x, *, window, **kw: x.rolling(window=window, **kw).std()
        if op == 'var':
            return lambda x, *, window, **kw: x.rolling(window=window, **kw).var()
        if op == 'min':
            return lambda x, *, window, **kw: x.rolling(window=window, **kw).min()
        if op == 'max':
            return lambda x, *, window, **kw: x.rolling(window=window, **kw).max()
        if op == 'sum':
            return lambda x, *, window, **kw: x.rolling(window=window, **kw).sum()
        raise ValueError(f"Unsupported rolling operation: {op}")

    def _vectorbt_rolling_func(op: str) -> Optional[Callable]:
        if not VECTORBT_AVAILABLE:
            return None
        mapping = {
            'mean': rolling_mean,
            'std': rolling_std,
            'var': rolling_var,
            'min': rolling_min,
            'max': rolling_max,
            'sum': rolling_sum,
        }
        func = mapping.get(op)
        return func

    op_name = (operation_name or 'mean').lower()
    vbt_func = _vectorbt_rolling_func(op_name)
    func = vbt_func if vbt_func is not None else _pandas_rolling_func(op_name)

    for window in window_sizes:
        batcher.add_operation(
            f"{op_name}_{window}",
            func,
            *(data,),
            window=window,
            priority=1,
            memory_weight=2.0
        )

    return batcher.execute_batch()

def batch_technical_indicators(data: pd.DataFrame,
                             indicators: List[Tuple[str, Callable, dict]],
                             batcher: Optional[VectorBTOperationBatcher] = None) -> Dict[str, Any]:
    """
    Batch multiple technical indicator calculations.
    
    Args:
        data: Input OHLCV data
        indicators: List of (name, function, kwargs) tuples
        batcher: Optional batcher instance
        
    Returns:
        Dictionary of indicator results
    """
    if batcher is None:
        batcher = VectorBTOperationBatcher()
    
    for name, func, kwargs in indicators:
        batcher.add_operation(
            name, func, data,
            priority=2,
            memory_weight=1.5,
            **kwargs
        )
    
    return batcher.execute_batch()

# Global batcher instance for convenience
_global_batcher = None

def get_global_batcher() -> VectorBTOperationBatcher:
    """Get the global VectorBT operation batcher instance."""
    global _global_batcher
    if _global_batcher is None:
        _global_batcher = VectorBTOperationBatcher()
    return _global_batcher

def clear_global_batcher() -> None:
    """Clear the global batcher and its queue."""
    global _global_batcher
    if _global_batcher is not None:
        _global_batcher.operations_queue.clear()
        _global_batcher.results_cache.clear()
