"""
Vectorized Processing Core for Training Pipeline Optimization.

This module provides comprehensive vectorized processing utilities optimized for
machine learning workflows, including matrix operations, memory management,
and data optimization for maximum performance.
"""

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Iterator
from contextlib import contextmanager
import gc
import logging
from pathlib import Path
import psutil
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from collections import deque
import asyncio
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineExecutionMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    HYBRID = "hybrid"


class PipelineStageStatus(Enum):
    """Pipeline stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Represents a single stage in the processing pipeline."""
    name: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: PipelineStageStatus = PipelineStageStatus.PENDING
    execution_time: float = 0.0
    memory_usage: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    output: Any = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Stage name cannot be empty")


@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
    success: bool
    total_time: float
    memory_peak: float
    stages_completed: int
    stages_failed: int
    stage_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OptimizedPipelineExecutor:
    """Optimized pipeline executor with intelligent scheduling and resource management."""

    def __init__(self, max_concurrent_stages: int = 4, enable_memory_tracking: bool = True,
                 enable_performance_monitoring: bool = True):
        """Initialize optimized pipeline executor.

        Args:
            max_concurrent_stages: Maximum number of stages to execute concurrently
            enable_memory_tracking: Whether to track memory usage per stage
            enable_performance_monitoring: Whether to monitor performance metrics
        """
        self.max_concurrent_stages = max_concurrent_stages
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_performance_monitoring = enable_performance_monitoring

        self.stages: Dict[str, PipelineStage] = {}
        self.execution_queue = deque()
        self.completed_stages = set()
        self.failed_stages = set()

        self.logger = logging.getLogger(f"{__name__}.OptimizedPipelineExecutor")

        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0.0
        }

    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline."""
        if stage.name in self.stages:
            raise ValueError(f"Stage '{stage.name}' already exists")

        self.stages[stage.name] = stage
        self.logger.debug(f"📝 Added pipeline stage: {stage.name}")

    def validate_pipeline(self) -> List[str]:
        """Validate pipeline configuration and dependencies."""
        errors = []

        # Check for missing dependencies
        for stage_name, stage in self.stages.items():
            for dep in stage.dependencies:
                if dep not in self.stages:
                    errors.append(f"Stage '{stage_name}' depends on missing stage '{dep}'")

        # Check for circular dependencies
        visited = set()
        recursion_stack = set()

        def has_cycle(stage_name: str) -> bool:
            visited.add(stage_name)
            recursion_stack.add(stage_name)

            for dep in self.stages[stage_name].dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in recursion_stack:
                    return True

            recursion_stack.remove(stage_name)
            return False

        for stage_name in self.stages:
            if stage_name not in visited:
                if has_cycle(stage_name):
                    errors.append(f"Circular dependency detected involving '{stage_name}'")

        return errors

    def get_execution_order(self) -> List[str]:
        """Determine optimal execution order based on dependencies."""
        # Simple topological sort
        result = []
        visited = set()
        temp_visited = set()

        def visit(stage_name: str):
            if stage_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {stage_name}")
            if stage_name in visited:
                return

            temp_visited.add(stage_name)

            # Visit dependencies first
            for dep in self.stages[stage_name].dependencies:
                visit(dep)

            temp_visited.remove(stage_name)
            visited.add(stage_name)
            result.append(stage_name)

        # Visit all stages
        for stage_name in self.stages:
            if stage_name not in visited:
                visit(stage_name)

        return result

    async def execute_async(self, execution_mode: PipelineExecutionMode = PipelineExecutionMode.HYBRID) -> PipelineExecutionResult:
        """Execute pipeline asynchronously with optimization."""
        start_time = time.time()
        peak_memory = 0

        # Validate pipeline
        validation_errors = self.validate_pipeline()
        if validation_errors:
            return PipelineExecutionResult(
                success=False,
                total_time=time.time() - start_time,
                memory_peak=peak_memory,
                stages_completed=0,
                stages_failed=0,
                errors=validation_errors
            )

        # Get execution order
        execution_order = self.get_execution_order()
        self.logger.info(f"🚀 Executing pipeline with {len(execution_order)} stages in {execution_mode.value} mode")

        # Execute based on mode
        if execution_mode == PipelineExecutionMode.SEQUENTIAL:
            result = await self._execute_sequential(execution_order, start_time)
        elif execution_mode == PipelineExecutionMode.PARALLEL:
            result = await self._execute_parallel(execution_order, start_time)
        elif execution_mode == PipelineExecutionMode.ASYNC:
            result = await self._execute_async_mode(execution_order, start_time)
        else:  # HYBRID
            result = await self._execute_hybrid(execution_order, start_time)

        # Update execution stats
        self.execution_stats['total_executions'] += 1
        if result.success:
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1

        self.execution_stats['average_execution_time'] = (
            (self.execution_stats['average_execution_time'] * (self.execution_stats['total_executions'] - 1)) +
            result.total_time
        ) / self.execution_stats['total_executions']

        if result.memory_peak > self.execution_stats['peak_memory_usage']:
            self.execution_stats['peak_memory_usage'] = result.memory_peak

        return result

    async def _execute_sequential(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute stages sequentially."""
        result = PipelineExecutionResult(
            success=True,
            total_time=0,
            memory_peak=0,
            stages_completed=0,
            stages_failed=0
        )

        for stage_name in execution_order:
            stage = self.stages[stage_name]
            stage.status = PipelineStageStatus.RUNNING

            try:
                stage_start = time.time()
                stage_start_memory = psutil.virtual_memory().percent if self.enable_memory_tracking else 0

                # Execute stage
                if asyncio.iscoroutinefunction(stage.func):
                    stage.output = await stage.func(*stage.args, **stage.kwargs)
                else:
                    stage.output = stage.func(*stage.args, **stage.kwargs)

                stage.execution_time = time.time() - stage_start
                stage.memory_usage = (
                    psutil.virtual_memory().percent - stage_start_memory
                ) if self.enable_memory_tracking else 0

                stage.status = PipelineStageStatus.COMPLETED
                result.stages_completed += 1
                result.stage_results[stage_name] = stage.output

                self.logger.debug(f"✅ Stage '{stage_name}' completed in {stage.execution_time:.2f}s")

            except Exception as e:
                stage.status = PipelineStageStatus.FAILED
                stage.error = str(e)
                result.stages_failed += 1
                result.errors.append(f"Stage '{stage_name}' failed: {e}")
                result.success = False

                # Retry logic
                if stage.retry_count < stage.max_retries:
                    stage.retry_count += 1
                    self.logger.warning(f"🔄 Retrying stage '{stage_name}' (attempt {stage.retry_count})")
                    # Reset to pending for retry
                    stage.status = PipelineStageStatus.PENDING
                    result.stages_failed -= 1  # Don't count as failed yet
                    continue

                self.logger.error(f"❌ Stage '{stage_name}' failed permanently: {e}")

        result.total_time = time.time() - start_time
        return result

    async def _execute_parallel(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute independent stages in parallel."""
        # Group stages by dependency level
        dependency_levels = self._group_by_dependency_level(execution_order)

        result = PipelineExecutionResult(
            success=True,
            total_time=0,
            memory_peak=0,
            stages_completed=0,
            stages_failed=0
        )

        for level in dependency_levels:
            if not level:
                continue

            # Execute stages in this level in parallel
            tasks = []
            for stage_name in level:
                task = asyncio.create_task(self._execute_stage(stage_name))
                tasks.append(task)

            # Wait for all stages in this level to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, res in enumerate(results):
                stage_name = level[i]
                if isinstance(res, Exception):
                    self.stages[stage_name].status = PipelineStageStatus.FAILED
                    self.stages[stage_name].error = str(res)
                    result.errors.append(f"Stage '{stage_name}' failed: {res}")
                    result.stages_failed += 1
                    result.success = False
                else:
                    result.stages_completed += 1
                    result.stage_results[stage_name] = res

        result.total_time = time.time() - start_time
        return result

    async def _execute_async_mode(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute all stages asynchronously with full concurrency."""
        # This is a simplified version - in practice, you'd want more sophisticated
        # dependency management for full async execution
        return await self._execute_parallel(execution_order, start_time)

    async def _execute_hybrid(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute pipeline using hybrid sequential-parallel approach."""
        # Analyze stage characteristics to decide execution strategy
        cpu_bound_stages = []
        io_bound_stages = []
        memory_intensive_stages = []

        for stage_name in execution_order:
            stage = self.stages[stage_name]
            # Simple heuristic based on function name (could be enhanced)
            if 'load' in stage_name.lower() or 'save' in stage_name.lower():
                io_bound_stages.append(stage_name)
            elif 'matrix' in stage_name.lower() or 'compute' in stage_name.lower():
                cpu_bound_stages.append(stage_name)
            else:
                memory_intensive_stages.append(stage_name)

        # Execute IO-bound stages first in parallel
        if io_bound_stages:
            await self._execute_parallel(io_bound_stages, start_time)

        # Execute CPU-bound stages in parallel
        if cpu_bound_stages:
            await self._execute_parallel(cpu_bound_stages, start_time)

        # Execute remaining stages sequentially
        remaining = [s for s in execution_order if s not in io_bound_stages + cpu_bound_stages]
        if remaining:
            result = await self._execute_sequential(remaining, start_time)
            return result

        # If all stages were executed in parallel groups, create a combined result
        return PipelineExecutionResult(
            success=True,
            total_time=time.time() - start_time,
            memory_peak=0,  # Would need to track this properly
            stages_completed=len(execution_order),
            stages_failed=0,
            stage_results={name: self.stages[name].output for name in execution_order}
        )

    def _group_by_dependency_level(self, execution_order: List[str]) -> List[List[str]]:
        """Group stages by dependency level for parallel execution."""
        levels = []
        completed = set()

        while len(completed) < len(execution_order):
            current_level = []

            for stage_name in execution_order:
                if stage_name in completed:
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = all(dep in completed for dep in self.stages[stage_name].dependencies)

                if deps_satisfied:
                    current_level.append(stage_name)

            if not current_level:
                # No stages can be executed (circular dependency or error)
                break

            levels.append(current_level)
            completed.update(current_level)

        return levels

    async def _execute_stage(self, stage_name: str) -> Any:
        """Execute a single stage asynchronously."""
        stage = self.stages[stage_name]
        stage.status = PipelineStageStatus.RUNNING

        try:
            stage_start = time.time()

            # Execute stage
            if asyncio.iscoroutinefunction(stage.func):
                result = await stage.func(*stage.args, **stage.kwargs)
            else:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, stage.func, *stage.args, stage.kwargs)

            stage.execution_time = time.time() - stage_start
            stage.output = result
            stage.status = PipelineStageStatus.COMPLETED

            self.logger.debug(f"✅ Stage '{stage_name}' completed in {stage.execution_time:.2f}s")
            return result

        except Exception as e:
            stage.status = PipelineStageStatus.FAILED
            stage.error = str(e)
            raise e

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline execution statistics."""
        return {
            'total_stages': len(self.stages),
            'execution_stats': self.execution_stats,
            'stage_status': {name: stage.status.value for name, stage in self.stages.items()},
            'stage_performance': {
                name: {
                    'execution_time': stage.execution_time,
                    'memory_usage': stage.memory_usage,
                    'retry_count': stage.retry_count
                }
                for name, stage in self.stages.items()
            }
        }

    def reset_pipeline(self):
        """Reset pipeline state for re-execution."""
        for stage in self.stages.values():
            stage.status = PipelineStageStatus.PENDING
            stage.execution_time = 0.0
            stage.memory_usage = 0.0
            stage.retry_count = 0
            stage.output = None
            stage.error = None

        self.completed_stages.clear()
        self.failed_stages.clear()
        self.execution_queue.clear()


class VectorizedProcessingCore:
    """Core class for vectorized processing operations with memory optimization."""

    def __init__(self, chunk_size: int = 50000, max_memory_gb: float = 8.0, enable_gpu: bool = True):
        """Initialize vectorized processing core.

        Args:
            chunk_size: Default chunk size for processing
            max_memory_gb: Maximum memory usage in GB
            enable_gpu: Whether to use GPU acceleration
        """
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.enable_gpu = enable_gpu

        # Initialize M1 optimizations if available
        try:
            from .m1_gpu_utils import get_m1_gpu_manager
            from .m1_memory_optimizer import get_m1_memory_optimizer
            from .m1_cpu_optimizer import get_m1_cpu_optimizer

            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            self.m1_available = True
        except ImportError:
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.m1_available = False

        self.logger = logger.getChild('VectorizedProcessingCore')
        self.logger.info("🔧 Vectorized Processing Core initialized")

        # Pipeline optimization components
        self.pipeline_executor = OptimizedPipelineExecutor(
            max_concurrent_stages=min(mp.cpu_count(), 8),  # M1 optimized
            enable_memory_tracking=True,
            enable_performance_monitoring=True
        )

    @contextmanager
    def memory_checkpoint(self, operation_name: str = "unknown"):
        """Context manager for memory monitoring during operations."""
        if self.m1_memory_optimizer:
            with self.m1_memory_optimizer.memory_checkpoint(operation_name):
                yield
        else:
            start_memory = psutil.virtual_memory().percent if psutil else 0
            start_time = time.time()
            try:
                yield
            finally:
                end_memory = psutil.virtual_memory().percent if psutil else 0
                duration = time.time() - start_time
                memory_delta = end_memory - start_memory
                self.logger.debug(
                    f"📊 {operation_name}: {duration:.2f}s, memory Δ: {memory_delta:+.1f}%"
                )

    def optimize_dataframe_for_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        with self.memory_checkpoint("dataframe_optimization"):
            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=['object']):
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')

            # Convert numeric columns to optimal dtypes
            for col in df.select_dtypes(include=[np.number]):
                if df[col].dtype == np.float64:
                    # Check if float32 is sufficient
                    if (df[col].max() < np.finfo(np.float32).max and
                        df[col].min() > np.finfo(np.float32).min):
                        df[col] = df[col].astype(np.float32)
                elif df[col].dtype == np.int64:
                    # Check if smaller integer type is sufficient
                    if df[col].max() < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)

            # Enhanced datetime dtype detection and optimization
            for col in df.columns:
                # Check for various datetime-related dtypes
                col_dtype = df[col].dtype

                # Handle datetime64 dtypes
                if pd.api.types.is_datetime64_any_dtype(col_dtype):
                    # Convert to consistent datetime64[ns] if not already
                    if col_dtype != 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(None)
                    # Consider converting to datetime64[s] for memory efficiency if precision allows
                    if df[col].dt.microsecond.eq(0).all() and df[col].dt.nanosecond.eq(0).all():
                        if df[col].dt.second.eq(0).all():
                            df[col] = df[col].astype('datetime64[s]')  # Second precision
                        else:
                            df[col] = df[col].astype('datetime64[ms]')  # Millisecond precision

                # Handle timedelta dtypes
                elif pd.api.types.is_timedelta64_dtype(col_dtype):
                    # Convert to consistent timedelta64[ns] if not already
                    if col_dtype != 'timedelta64[ns]':
                        df[col] = pd.to_timedelta(df[col])

                # Handle object columns that might contain datetime strings
                elif col_dtype == 'object':
                    # Sample values to detect datetime patterns
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        try:
                            # Try to infer datetime format
                            pd.to_datetime(sample_values, infer_datetime_format=True, errors='coerce')
                            # If successful, convert the column
                            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                        except (ValueError, TypeError):
                            pass  # Not datetime-like, leave as object

            return df

    def vectorized_rolling_features(self, data: pd.DataFrame,
                                  windows: List[int] = [5, 10, 20, 50],
                                  features: List[str] = None) -> pd.DataFrame:
        """Create vectorized rolling features.

        Args:
            data: Input DataFrame
            windows: Rolling window sizes
            features: Columns to create features for

        Returns:
            DataFrame with rolling features
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        with self.memory_checkpoint("rolling_features"):
            result_dfs = []

            for window in windows:
                window_features = {}
                for col in features:
                    if col in data.columns:
                        series = data[col]

                        # Vectorized rolling calculations
                        window_features[f'{col}_rolling_mean_{window}'] = series.rolling(window=window, min_periods=1).mean()
                        window_features[f'{col}_rolling_std_{window}'] = series.rolling(window=window, min_periods=1).std()
                        window_features[f'{col}_rolling_min_{window}'] = series.rolling(window=window, min_periods=1).min()
                        window_features[f'{col}_rolling_max_{window}'] = series.rolling(window=window, min_periods=1).max()
                        window_features[f'{col}_rolling_skew_{window}'] = series.rolling(window=window, min_periods=3).skew()
                        window_features[f'{col}_rolling_kurt_{window}'] = series.rolling(window=window, min_periods=4).kurt()

                result_dfs.append(pd.DataFrame(window_features))

            # Combine all features efficiently
            if result_dfs:
                combined = pd.concat(result_dfs, axis=1)
                return pd.concat([data, combined], axis=1)
            return data

    def matrix_correlation_analysis(self, data: pd.DataFrame,
                                   method: str = 'pearson') -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute matrix-based correlation analysis.

        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Tuple of correlation matrix and feature importance scores
        """
        with self.memory_checkpoint("correlation_analysis"):
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.shape[1] < 2:
                return np.array([[1.0]]), pd.DataFrame()

            # Vectorized correlation computation
            if method == 'pearson':
                corr_matrix = numeric_data.corr().values
            elif method == 'spearman':
                corr_matrix = numeric_data.corr(method='spearman').values
            else:  # kendall
                corr_matrix = numeric_data.corr(method='kendall').values

            # Compute feature importance based on correlation strength
            feature_importance = pd.DataFrame({
                'feature': numeric_data.columns,
                'mean_abs_corr': np.abs(corr_matrix).mean(axis=1),
                'max_corr': np.abs(corr_matrix).max(axis=1),
                'corr_std': np.abs(corr_matrix).std(axis=1)
            })

            return corr_matrix, feature_importance

    def chunked_matrix_operations(self, data: pd.DataFrame,
                                operation_func: Callable[[pd.DataFrame], T],
                                chunk_size: Optional[int] = None) -> List[T]:
        """Perform chunked matrix operations with memory management.

        Args:
            data: Input DataFrame
            operation_func: Function to apply to each chunk
            chunk_size: Size of each chunk

        Returns:
            List of operation results
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        with self.memory_checkpoint("chunked_operations"):
            results = []
            total_rows = len(data)

            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = data.iloc[start_idx:end_idx]

                # Apply operation to chunk
                result = operation_func(chunk)
                results.append(result)

                # Memory cleanup between chunks
                if len(results) % 10 == 0:  # Every 10 chunks
                    if self.m1_memory_optimizer:
                        self.m1_memory_optimizer.optimize_memory()
                    else:
                        gc.collect()

            return results

    def parallel_feature_engineering(self, data: pd.DataFrame,
                                   feature_functions: List[Callable[[pd.DataFrame], pd.Series]],
                                   max_workers: Optional[int] = None) -> pd.DataFrame:
        """Parallel feature engineering with optimized processing.

        Args:
            data: Input DataFrame
            feature_functions: List of feature engineering functions
            max_workers: Maximum number of parallel workers

        Returns:
            DataFrame with engineered features
        """
        if not feature_functions:
            return data

        with self.memory_checkpoint("parallel_feature_engineering"):
            if self.m1_cpu_optimizer and max_workers is None:
                max_workers = self.m1_cpu_optimizer.get_optimal_workers_for_task("cpu_bound")

            # Use parallel processing if beneficial
            if max_workers > 1 and len(feature_functions) > 1:
                if self.m1_cpu_optimizer:
                    results = self.m1_cpu_optimizer.parallel_process(
                        feature_functions,
                        lambda func: func(data),
                        task_type="cpu_bound"
                    )
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(lambda func: func(data), feature_functions))
            else:
                results = [func(data) for func in feature_functions]

            # Combine results
            feature_df = pd.concat(results, axis=1)
            return pd.concat([data, feature_df], axis=1)

    def optimized_train_test_split(self, X: Union[pd.DataFrame, np.ndarray],
                                 y: Union[pd.Series, np.ndarray],
                                 test_size: float = 0.2,
                                 shuffle: bool = False,
                                 stratify: Optional[np.ndarray] = None) -> Tuple:
        """Optimized train-test split with memory efficiency.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            shuffle: Whether to shuffle data
            stratify: Stratification array

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        with self.memory_checkpoint("train_test_split"):
            n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
            n_test = int(n_samples * test_size)
            n_train = n_samples - n_test

            if shuffle:
                indices = np.random.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            # Efficient indexing based on data type
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_indices]
                X_test = X.iloc[test_indices]
            else:
                X_train = X[train_indices]
                X_test = X[test_indices]

            if isinstance(y, pd.Series):
                y_train = y.iloc[train_indices]
                y_test = y.iloc[test_indices]
            else:
                y_train = y[train_indices]
                y_test = y[test_indices]

            return X_train, X_test, y_train, y_test

    def gpu_accelerated_matrix_ops(self, matrix_a: np.ndarray,
                                 matrix_b: Optional[np.ndarray] = None,
                                 operation: str = "multiply") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """GPU-accelerated matrix operations.

        Args:
            matrix_a: First matrix
            matrix_b: Second matrix (if needed)
            operation: Operation type ('multiply', 'add', 'subtract', 'eigen', 'svd')

        Returns:
            Result matrix
        """
        if not self.enable_gpu or not self.m1_gpu_manager:
            # Fallback to CPU operations
            return self._cpu_matrix_ops(matrix_a, matrix_b, operation)

        with self.memory_checkpoint("gpu_matrix_ops"):
            try:
                # Convert to tensors
                tensor_a = self.m1_gpu_manager.to_device(matrix_a, "matrix_mult")

                if matrix_b is not None:
                    tensor_b = self.m1_gpu_manager.to_device(matrix_b, "matrix_mult")
                else:
                    tensor_b = None

                # Perform operation
                if operation == "multiply":
                    if tensor_b is not None:
                        result = self.m1_gpu_manager.matrix_multiply_mps(tensor_a, tensor_b)
                    else:
                        result = tensor_a @ tensor_a
                elif operation == "add":
                    result = tensor_a + tensor_b
                elif operation == "subtract":
                    result = tensor_a - tensor_b
                elif operation == "eigen":
                    eigenvals = torch.linalg.eigvals(tensor_a)
                    result = eigenvals
                elif operation == "svd":
                    U, S, V = torch.linalg.svd(tensor_a)
                    result = (U, S, V)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                # Convert back to numpy
                if isinstance(result, tuple):
                    return tuple(r.cpu().numpy() for r in result)
                else:
                    return result.cpu().numpy()

            except Exception as e:
                self.logger.warning(f"GPU operation failed: {e}, falling back to CPU")
                return self._cpu_matrix_ops(matrix_a, matrix_b, operation)

    def _cpu_matrix_ops(self, matrix_a: np.ndarray,
                       matrix_b: Optional[np.ndarray],
                       operation: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """CPU fallback for matrix operations."""
        if operation == "multiply":
            if matrix_b is not None:
                return np.dot(matrix_a, matrix_b)
            else:
                return np.dot(matrix_a, matrix_a)
        elif operation == "add":
            return matrix_a + matrix_b
        elif operation == "subtract":
            return matrix_a - matrix_b
        elif operation == "eigen":
            return np.linalg.eigvals(matrix_a)
        elif operation == "svd":
            return np.linalg.svd(matrix_a)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def memory_efficient_groupby(self, df: pd.DataFrame,
                               group_cols: List[str],
                               agg_funcs: Dict[str, str],
                               chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Memory-efficient groupby operations.

        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_funcs: Aggregation functions per column
            chunk_size: Processing chunk size

        Returns:
            Aggregated DataFrame
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        with self.memory_checkpoint("memory_efficient_groupby"):
            if len(df) <= chunk_size:
                # Single operation
                return df.groupby(group_cols).agg(agg_funcs)

            # Chunked processing
            results = []
            for start_idx in range(0, len(df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx]

                chunk_result = chunk.groupby(group_cols).agg(agg_funcs)
                results.append(chunk_result)

                # Memory cleanup
                if self.m1_memory_optimizer and len(results) % 5 == 0:
                    self.m1_memory_optimizer.optimize_memory()

            # Combine results
            if results:
                final_result = pd.concat(results)
                # Re-aggregate to handle overlapping groups
                return final_result.groupby(level=group_cols).agg(agg_funcs)

            return pd.DataFrame()

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        stats = {
            'chunk_size': self.chunk_size,
            'max_memory_gb': self.max_memory_gb,
            'gpu_enabled': self.enable_gpu,
            'm1_optimizations': self.m1_available
        }

        if psutil:
            memory = psutil.virtual_memory()
            stats.update({
                'current_memory_percent': memory.percent,
                'available_memory_gb': memory.available / (1024**3),
                'total_memory_gb': memory.total / (1024**3)
            })

        if self.m1_memory_optimizer:
            memory_report = self.m1_memory_optimizer.get_memory_report()
            stats['memory_efficiency'] = memory_report.get('memory_efficiency', 0.0)

        return stats

    def create_optimized_pipeline(self, stages_config: List[Dict[str, Any]]) -> OptimizedPipelineExecutor:
        """Create an optimized processing pipeline from configuration.

        Args:
            stages_config: List of stage configurations with format:
                {
                    'name': str,
                    'func': Callable,
                    'args': Tuple = (),
                    'kwargs': Dict = {},
                    'dependencies': List[str] = []
                }

        Returns:
            Configured OptimizedPipelineExecutor
        """
        executor = OptimizedPipelineExecutor()

        for config in stages_config:
            stage = PipelineStage(
                name=config['name'],
                func=config['func'],
                args=config.get('args', ()),
                kwargs=config.get('kwargs', {}),
                dependencies=config.get('dependencies', [])
            )
            executor.add_stage(stage)

        return executor

    def execute_ml_pipeline(self, data: pd.DataFrame,
                          pipeline_config: List[Dict[str, Any]],
                          execution_mode: PipelineExecutionMode = PipelineExecutionMode.HYBRID) -> PipelineExecutionResult:
        """Execute a complete ML processing pipeline with optimization.

        Args:
            data: Input DataFrame
            pipeline_config: Pipeline stage configuration
            execution_mode: Execution mode for the pipeline

        Returns:
            Pipeline execution results
        """
        # Create pipeline with data passed to each stage
        stages_config = []
        for config in pipeline_config:
            # Add data as first argument to each stage function
            stage_config = config.copy()
            stage_config['args'] = (data,) + stage_config.get('args', ())
            stages_config.append(stage_config)

        pipeline = self.create_optimized_pipeline(stages_config)

        # Execute pipeline
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                pipeline.execute_async(execution_mode)
            )

            self.logger.info(
                f"📊 Pipeline execution completed: {result.stages_completed}/{result.stages_completed + result.stages_failed} stages successful "
                f"in {result.total_time:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return PipelineExecutionResult(
                success=False,
                total_time=0,
                memory_peak=0,
                stages_completed=0,
                stages_failed=len(pipeline_config),
                errors=[str(e)]
            )

    def optimize_pipeline_execution(self, pipeline_config: List[Dict[str, Any]],
                                  data_sample: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and optimize pipeline execution strategy.

        Args:
            pipeline_config: Pipeline configuration
            data_sample: Sample data for analysis

        Returns:
            Optimization recommendations
        """
        analysis = {
            'recommended_mode': PipelineExecutionMode.SEQUENTIAL.value,
            'estimated_parallel_speedup': 1.0,
            'memory_requirements': {},
            'bottleneck_analysis': {},
            'optimization_suggestions': []
        }

        # Analyze stage characteristics
        stage_complexity = {}
        for config in pipeline_config:
            stage_name = config['name']

            # Estimate complexity based on function characteristics
            if 'load' in stage_name.lower() or 'save' in stage_name.lower():
                complexity = 'io_bound'
            elif 'matrix' in stage_name.lower() or 'compute' in stage_name.lower():
                complexity = 'cpu_bound'
            elif 'feature' in stage_name.lower() or 'transform' in stage_name.lower():
                complexity = 'memory_bound'
            else:
                complexity = 'general'

            stage_complexity[stage_name] = complexity

        # Count stage types
        complexity_counts = {}
        for complexity in stage_complexity.values():
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Recommend execution mode based on stage characteristics
        total_stages = len(pipeline_config)
        io_stages = complexity_counts.get('io_bound', 0)
        cpu_stages = complexity_counts.get('cpu_bound', 0)

        if io_stages > total_stages * 0.5:
            # Mostly IO-bound - use parallel execution
            analysis['recommended_mode'] = PipelineExecutionMode.PARALLEL.value
            analysis['estimated_parallel_speedup'] = min(io_stages, self.pipeline_executor.max_concurrent_stages)
        elif cpu_stages > total_stages * 0.3:
            # Significant CPU-bound work - use hybrid execution
            analysis['recommended_mode'] = PipelineExecutionMode.HYBRID.value
            analysis['estimated_parallel_speedup'] = min(cpu_stages * 0.7, self.pipeline_executor.max_concurrent_stages)
        else:
            # General workload - use sequential with memory optimization
            analysis['recommended_mode'] = PipelineExecutionMode.SEQUENTIAL.value
            analysis['estimated_parallel_speedup'] = 1.0

        # Memory requirements analysis
        try:
            sample_memory = data_sample.memory_usage(deep=True).sum() / (1024**2)  # MB
            analysis['memory_requirements'] = {
                'sample_data_size_mb': sample_memory,
                'estimated_pipeline_memory_mb': sample_memory * 2,  # Rough estimate
                'recommended_chunk_size': max(1000, int(len(data_sample) / 10))
            }
        except Exception as e:
            self.logger.debug(f"Memory analysis failed: {e}")

        # Generate optimization suggestions
        if analysis['estimated_parallel_speedup'] > 2.0:
            analysis['optimization_suggestions'].append(
                "High parallel speedup potential - consider using PARALLEL execution mode"
            )

        if self.m1_available:
            analysis['optimization_suggestions'].append(
                "M1 optimizations available - GPU acceleration recommended for matrix operations"
            )

        if self.memory_optimizer and analysis['memory_requirements'].get('estimated_pipeline_memory_mb', 0) > 1000:
            analysis['optimization_suggestions'].append(
                "Large memory requirements - chunked processing recommended"
            )

        return analysis


# Global instance for easy access
_vectorized_core = None

def get_vectorized_processing_core() -> VectorizedProcessingCore:
    """Get global vectorized processing core instance."""
    global _vectorized_core
    if _vectorized_core is None:
        _vectorized_core = VectorizedProcessingCore()
    return _vectorized_core


# Convenience functions
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame for processing."""
    core = get_vectorized_processing_core()
    return core.optimize_dataframe_for_processing(df)


def vectorized_rolling_features(data: pd.DataFrame,
                              windows: List[int] = None,
                              features: List[str] = None) -> pd.DataFrame:
    """Create vectorized rolling features."""
    if windows is None:
        windows = [5, 10, 20, 50]
    core = get_vectorized_processing_core()
    return core.vectorized_rolling_features(data, windows, features)


def matrix_correlation_analysis(data: pd.DataFrame,
                              method: str = 'pearson') -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute matrix-based correlation analysis."""
    core = get_vectorized_processing_core()
    return core.matrix_correlation_analysis(data, method)


def parallel_feature_engineering(data: pd.DataFrame,
                               feature_functions: List[Callable[[pd.DataFrame], pd.Series]],
                               max_workers: Optional[int] = None) -> pd.DataFrame:
    """Parallel feature engineering."""
    core = get_vectorized_processing_core()
    return core.parallel_feature_engineering(data, feature_functions, max_workers)


def gpu_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU-accelerated matrix multiplication."""
    core = get_vectorized_processing_core()
    return core.gpu_accelerated_matrix_ops(a, b, "multiply")


def create_ml_pipeline(stages_config: List[Dict[str, Any]]) -> OptimizedPipelineExecutor:
    """Create an optimized ML processing pipeline."""
    core = get_vectorized_processing_core()
    return core.create_optimized_pipeline(stages_config)


def execute_ml_pipeline(data: pd.DataFrame,
                       pipeline_config: List[Dict[str, Any]],
                       execution_mode: PipelineExecutionMode = PipelineExecutionMode.HYBRID) -> PipelineExecutionResult:
    """Execute a complete ML processing pipeline with optimization."""
    core = get_vectorized_processing_core()
    return core.execute_ml_pipeline(data, pipeline_config, execution_mode)


def optimize_pipeline_config(pipeline_config: List[Dict[str, Any]],
                           data_sample: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and optimize pipeline execution strategy."""
    core = get_vectorized_processing_core()
    return core.optimize_pipeline_execution(pipeline_config, data_sample)


def get_pipeline_executor() -> OptimizedPipelineExecutor:
    """Get the global pipeline executor instance."""
    core = get_vectorized_processing_core()
    return core.pipeline_executor
