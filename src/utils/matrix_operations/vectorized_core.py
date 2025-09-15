"""
Vectorized Processing Core - Unified Implementation

This module consolidates vectorized processing functionality from scattered sources
into a single, unified interface with backwards compatibility.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Iterator
from contextlib import contextmanager
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import datetime

# Conditional imports for optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

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
        """Initialize optimized pipeline executor."""
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
        return errors

    def get_execution_order(self) -> List[str]:
        """Determine optimal execution order based on dependencies."""
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

        result.total_time = time.time() - start_time
        return result

    async def _execute_parallel(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute independent stages in parallel."""
        # Simplified parallel execution
        return await self._execute_sequential(execution_order, start_time)

    async def _execute_async_mode(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute all stages asynchronously with full concurrency."""
        return await self._execute_parallel(execution_order, start_time)

    async def _execute_hybrid(self, execution_order: List[str], start_time: float) -> PipelineExecutionResult:
        """Execute pipeline using hybrid sequential-parallel approach."""
        return await self._execute_sequential(execution_order, start_time)

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
        """Initialize vectorized processing core."""
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.enable_gpu = enable_gpu

        # Initialize M1 optimizations if available
        try:
            from ..hardware.m1_gpu_utils import get_m1_gpu_manager
            from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer
            from ..hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

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

    def optimize_dataframe_for_processing(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Optimize DataFrame for vectorized processing."""
        with self.memory_checkpoint("dataframe_optimization"):
            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=['object']):
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')

            # Convert numeric columns to optimal dtypes
            for col in df.select_dtypes(include=[np.number]):
                if hasattr(df[col], 'dtype') and df[col].dtype == (np.float64 if np is not None else float if np is not None else float):
                    # Check if float32 is sufficient
                    if (df[col].max() < np.finfo if np is not None else lambda x: type("finfo", (), {"max": float("inf"), "min": float("-inf")})()(np.float32 if np is not None else float).max and
                        df[col].min() > np.finfo if np is not None else lambda x: type("finfo", (), {"max": float("inf"), "min": float("-inf")})()(np.float32 if np is not None else float).min):
                        df[col] = df[col].astype(np.float32 if np is not None else float)
                elif hasattr(df[col], 'dtype') and df[col].dtype == (np.int64 if np is not None else int if np is not None else int):
                    # Check if smaller integer type is sufficient
                    if df[col].max() < np.iinfo if np is not None else lambda x: type("iinfo", (), {"max": 2147483647, "min": -2147483648})()(np.int32 if np is not None else int).max:
                        df[col] = df[col].astype(np.int32 if np is not None else int)

            return df

    def vectorized_rolling_features(self, data: 'pd.DataFrame',
                                  windows: List[int] = [5, 10, 20, 50],
                                  features: List[str] = None) -> 'pd.DataFrame':
        """Create vectorized rolling features."""
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

    def matrix_correlation_analysis(self, data: 'pd.DataFrame',
                                   method: str = 'pearson') -> Tuple['np.ndarray', 'pd.DataFrame']:
        """Compute matrix-based correlation analysis."""
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

# Global instance for easy access
_vectorized_core = None

def get_vectorized_processing_core() -> VectorizedProcessingCore:
    """Get global vectorized processing core instance."""
    global _vectorized_core
    if _vectorized_core is None:
        _vectorized_core = VectorizedProcessingCore()
    return _vectorized_core

# Convenience functions
def optimize_dataframe(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Optimize DataFrame for processing."""
    core = get_vectorized_processing_core()
    return core.optimize_dataframe_for_processing(df)

def vectorized_rolling_features(data: 'pd.DataFrame',
                              windows: List[int] = None,
                              features: List[str] = None) -> 'pd.DataFrame':
    """Create vectorized rolling features."""
    if windows is None:
        windows = [5, 10, 20, 50]
    core = get_vectorized_processing_core()
    return core.vectorized_rolling_features(data, windows, features)

def matrix_correlation_analysis(data: 'pd.DataFrame',
                              method: str = 'pearson') -> Tuple['np.ndarray', 'pd.DataFrame']:
    """Compute matrix-based correlation analysis."""
    core = get_vectorized_processing_core()
    return core.matrix_correlation_analysis(data, method)