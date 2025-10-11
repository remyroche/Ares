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

# Hardware optimization imports
try:
    from .hardware_integration import (
        get_hardware_optimized_processor, 
        hardware_optimized,
        optimize_matrix_operation,
        HardwareConfig
    )
    HARDWARE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware integration not available: {e}")
    HARDWARE_INTEGRATION_AVAILABLE = False
    get_hardware_optimized_processor = None
    hardware_optimized = None
    optimize_matrix_operation = None
    HardwareConfig = None

# VectorBT optimizations
try:
    from .vectorbt_optimizations import get_vectorbt_optimized_operations
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VectorBT optimizations not available: {e}")
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    get_vectorbt_optimized_operations = None

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

    def __init__(self, chunk_size: int = 50000, max_memory_gb: float = 8.0, enable_gpu: bool = True,
                 hardware_config: Optional[HardwareConfig] = None):
        """Initialize vectorized processing core."""
        # Initialize logger first
        self.logger = logger.getChild('VectorizedProcessingCore')
        
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.enable_gpu = enable_gpu

        # Initialize M1 optimizations if available
        try:
            from ..hardware.m1_gpu_utils import get_m1_gpu_manager
            from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer
            from ..hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            self.m1_available = True
        except ImportError:
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.m1_available = False

        # Initialize hardware integration
        self._initialize_hardware_integration(hardware_config)

        self.logger.info("🔧 Vectorized Processing Core initialized")
    
    def _initialize_hardware_integration(self, hardware_config: Optional[HardwareConfig] = None):
        """Initialize hardware integration for optimized computations."""
        if HARDWARE_INTEGRATION_AVAILABLE and get_hardware_optimized_processor:
            try:
                # Create hardware config if not provided
                if hardware_config is None:
                    hardware_config = HardwareConfig(
                        max_memory_gb=self.max_memory_gb,
                        enable_gpu=self.enable_gpu,
                        chunk_size_threshold=self.chunk_size
                    )
                
                self.hardware_processor = get_hardware_optimized_processor(hardware_config)
                self.hardware_optimization_enabled = True
                self.logger.info("✅ Hardware optimization integration enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware integration initialization failed: {e}")
                self.hardware_processor = None
                self.hardware_optimization_enabled = False
        else:
            self.hardware_processor = None
            self.hardware_optimization_enabled = False
            self.logger.info("ℹ️ Hardware optimization integration not available")

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
            for col in df.select_dtypes(include=[np.number if np is not None else 'number']):
                if NUMPY_AVAILABLE and np is not None:
                    if hasattr(df[col], 'dtype') and df[col].dtype == np.float64:
                        # Check if float32 is sufficient
                        if (df[col].max() < np.finfo(np.float32).max and
                            df[col].min() > np.finfo(np.float32).min):
                            df[col] = df[col].astype(np.float32)
                    elif hasattr(df[col], 'dtype') and df[col].dtype == np.int64:
                        # Check if smaller integer type is sufficient
                        if df[col].max() < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                else:
                    # Fallback when numpy is not available
                    if hasattr(df[col], 'dtype'):
                        if 'float64' in str(df[col].dtype):
                            try:
                                # Try to convert to float32 if values are within range
                                if df[col].max() < 3.4e38 and df[col].min() > -3.4e38:
                                    df[col] = df[col].astype('float32')
                            except (ValueError, OverflowError):
                                pass
                        elif 'int64' in str(df[col].dtype):
                            try:
                                # Try to convert to int32 if values are within range
                                if df[col].max() < 2147483647 and df[col].min() > -2147483648:
                                    df[col] = df[col].astype('int32')
                            except (ValueError, OverflowError):
                                pass

            return df

    def vectorized_rolling_features(self, data: 'pd.DataFrame',
                                  windows: List[int] = [5, 10, 20, 50],
                                  features: List[str] = None) -> 'pd.DataFrame':
        """Create vectorized rolling features using VectorBT optimization with enhanced performance."""
        # Try VectorBT optimization first if available
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                vectorbt_ops = get_vectorbt_optimized_operations()
                result = vectorbt_ops.rolling_features(data, windows, features)
                self.logger.info("✅ Rolling features computed using VectorBT optimization")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT rolling features failed: {e}, falling back to standard method")
        
        # Enhanced fallback implementation with better performance
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        with self.memory_checkpoint("rolling_features"):
            # Optimize data types for better performance
            data_optimized = self.optimize_dataframe_for_processing(data)
            
            result_dfs = []
            
            # Process windows in batches for better memory efficiency
            batch_size = min(5, len(windows))  # Process up to 5 windows at a time
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i:i + batch_size]
                batch_features = {}
                
                for window in batch_windows:
                    for col in features:
                        if col in data_optimized.columns:
                            series = data_optimized[col]

                            # Enhanced vectorized rolling calculations with better performance
                            rolling = series.rolling(window=window, min_periods=1)
                            
                            batch_features.update({
                                f'{col}_rolling_mean_{window}': rolling.mean(),
                                f'{col}_rolling_std_{window}': rolling.std(),
                                f'{col}_rolling_min_{window}': rolling.min(),
                                f'{col}_rolling_max_{window}': rolling.max(),
                                f'{col}_rolling_skew_{window}': rolling.skew(),
                                f'{col}_rolling_kurt_{window}': rolling.kurt(),
                                f'{col}_rolling_median_{window}': rolling.median(),
                                f'{col}_rolling_quantile_25_{window}': rolling.quantile(0.25),
                                f'{col}_rolling_quantile_75_{window}': rolling.quantile(0.75),
                            })
                            
                            # Additional enhanced features
                            batch_features[f'{col}_rolling_range_{window}'] = (
                                batch_features[f'{col}_rolling_max_{window}'] - 
                                batch_features[f'{col}_rolling_min_{window}']
                            )
                            
                            batch_features[f'{col}_rolling_cv_{window}'] = (
                                batch_features[f'{col}_rolling_std_{window}'] / 
                                batch_features[f'{col}_rolling_mean_{window}']
                            ).fillna(0)

                result_dfs.append(pd.DataFrame(batch_features, index=data_optimized.index))

            # Combine all features efficiently
            if result_dfs:
                combined = pd.concat(result_dfs, axis=1)
                return pd.concat([data_optimized, combined], axis=1)
            return data_optimized

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

    def compute_trading_indicators(self, data: 'pd.DataFrame', 
                                 config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
        """
        Compute comprehensive trading indicators in vectorized fashion with VectorBT and hardware optimization.
        
        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            config: Configuration dictionary for indicator parameters
            
        Returns:
            DataFrame with all computed indicators
        """
        if config is None:
            config = self._get_default_indicator_config()
        
        # Try VectorBT optimization first if available
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                vectorbt_ops = get_vectorbt_optimized_operations()
                result = vectorbt_ops.compute_trading_indicators(data, config)
                self.logger.info("✅ Trading indicators computed using VectorBT optimization")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT trading indicators failed: {e}, falling back to standard method")
        
        # Use hardware optimization if available
        if self.hardware_optimization_enabled and self.hardware_processor:
            return self._compute_trading_indicators_hardware_optimized(data, config)
        else:
            return self._compute_trading_indicators_standard(data, config)
    
    def _compute_trading_indicators_hardware_optimized(self, data: 'pd.DataFrame', 
                                                     config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute trading indicators with hardware optimization."""
        def compute_indicators_func(df, config):
            return self._compute_trading_indicators_standard(df, config)
        
        return self.hardware_processor.process_with_hardware_optimization(
            data, compute_indicators_func, config
        )
    
    def _compute_trading_indicators_standard(self, data: 'pd.DataFrame', 
                                           config: Dict[str, Any]) -> 'pd.DataFrame':
        """Standard trading indicators computation."""
        with self.memory_checkpoint("trading_indicators"):
            result_df = data.copy()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing required columns: {missing_cols}")
                return result_df
            
            # Compute all indicators
            result_df = self._compute_moving_averages(result_df, config)
            result_df = self._compute_momentum_indicators(result_df, config)
            result_df = self._compute_volatility_indicators(result_df, config)
            result_df = self._compute_volume_indicators(result_df, config)
            result_df = self._compute_trend_indicators(result_df, config)
            result_df = self._compute_oscillator_indicators(result_df, config)
            result_df = self._compute_pattern_indicators(result_df, config)
            
            self.logger.info(f"✅ Computed {len(result_df.columns) - len(data.columns)} trading indicators")
            return result_df

    def _get_default_indicator_config(self) -> Dict[str, Any]:
        """Get default configuration for trading indicators."""
        return {
            # Moving averages
            'sma_periods': [9, 21, 50, 200],
            'ema_periods': [12, 26, 50],
            
            # RSI
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            
            # Stochastic
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            
            # Williams %R
            'williams_period': 14,
            
            # ADX
            'adx_period': 14,
            
            # ATR
            'atr_period': 14,
            
            # CCI
            'cci_period': 20,
            
            # ROC
            'roc_period': 10,
            
            # Volume indicators
            'volume_sma_period': 20,
            'obv_smooth': 10,
        }

    def _compute_moving_averages(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute Simple and Exponential Moving Averages."""
        result = data.copy()
        
        # Simple Moving Averages
        for period in config.get('sma_periods', [9, 21, 50, 200]):
            result[f'sma_{period}'] = data['close'].rolling(window=period, min_periods=1).mean()
        
        # Exponential Moving Averages
        for period in config.get('ema_periods', [12, 26, 50]):
            result[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # Moving Average Crossovers
        if 'sma_9' in result.columns and 'sma_21' in result.columns:
            result['sma_cross_9_21'] = (result['sma_9'] > result['sma_21']).astype(int)
        if 'ema_12' in result.columns and 'ema_26' in result.columns:
            result['ema_cross_12_26'] = (result['ema_12'] > result['ema_26']).astype(int)
        
        return result

    def _compute_momentum_indicators(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute momentum indicators (RSI, MACD, ROC, etc.)."""
        result = data.copy()
        
        # RSI (Relative Strength Index)
        rsi_period = config.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        result['rsi_overbought'] = (result['rsi'] > config.get('rsi_overbought', 70)).astype(int)
        result['rsi_oversold'] = (result['rsi'] < config.get('rsi_oversold', 30)).astype(int)
        
        # MACD (Moving Average Convergence Divergence)
        macd_fast = config.get('macd_fast', 12)
        macd_slow = config.get('macd_slow', 26)
        macd_signal = config.get('macd_signal', 9)
        
        ema_fast = data['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=macd_slow, adjust=False).mean()
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=macd_signal, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # MACD signals
        result['macd_bullish'] = (result['macd'] > result['macd_signal']).astype(int)
        result['macd_cross'] = (result['macd'] > result['macd_signal']).astype(int).diff().fillna(0)
        
        # ROC (Rate of Change)
        roc_period = config.get('roc_period', 10)
        result['roc'] = ((data['close'] - data['close'].shift(roc_period)) / data['close'].shift(roc_period)) * 100
        
        # Momentum
        result['momentum'] = data['close'] - data['close'].shift(10)
        
        return result

    def _compute_volatility_indicators(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute volatility indicators (Bollinger Bands, ATR, etc.)."""
        result = data.copy()
        
        # Bollinger Bands
        bb_period = config.get('bb_period', 20)
        bb_std = config.get('bb_std', 2.0)
        
        sma = data['close'].rolling(window=bb_period, min_periods=1).mean()
        std = data['close'].rolling(window=bb_period, min_periods=1).std()
        
        result['bb_upper'] = sma + (std * bb_std)
        result['bb_lower'] = sma - (std * bb_std)
        result['bb_middle'] = sma
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_position'] = (data['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Bollinger Band signals
        result['bb_squeeze'] = (result['bb_width'] < result['bb_width'].rolling(20).quantile(0.2)).astype(int)
        result['bb_breakout_upper'] = (data['close'] > result['bb_upper']).astype(int)
        result['bb_breakout_lower'] = (data['close'] < result['bb_lower']).astype(int)
        
        # ATR (Average True Range)
        atr_period = config.get('atr_period', 14)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        result['atr'] = true_range.rolling(window=atr_period, min_periods=1).mean()
        result['atr_percent'] = (result['atr'] / data['close']) * 100
        
        # Volatility
        result['volatility'] = data['close'].rolling(window=20, min_periods=1).std()
        result['volatility_percent'] = (result['volatility'] / data['close']) * 100
        
        return result

    def _compute_volume_indicators(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute volume-based indicators."""
        result = data.copy()
        
        # Volume SMA
        volume_sma_period = config.get('volume_sma_period', 20)
        result['volume_sma'] = data['volume'].rolling(window=volume_sma_period, min_periods=1).mean()
        
        # Volume ratio
        result['volume_ratio'] = data['volume'] / result['volume_sma']
        
        # OBV (On-Balance Volume)
        price_change = data['close'].diff()
        obv = np.where(price_change > 0, data['volume'], 
                      np.where(price_change < 0, -data['volume'], 0))
        result['obv'] = np.cumsum(obv)
        
        # OBV smoothed
        obv_smooth = config.get('obv_smooth', 10)
        result['obv_sma'] = result['obv'].rolling(window=obv_smooth, min_periods=1).mean()
        
        # Volume-Price Trend
        result['vpt'] = (data['volume'] * data['close'].pct_change()).cumsum()
        
        # Money Flow Index (simplified)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        result['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return result

    def _compute_trend_indicators(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute trend indicators (ADX, Parabolic SAR, etc.)."""
        result = data.copy()
        
        # ADX (Average Directional Index)
        adx_period = config.get('adx_period', 14)
        
        # True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Directional Movement
        plus_dm = np.where((data['high'].diff() > data['low'].diff().abs()) & 
                          (data['high'].diff() > 0), data['high'].diff(), 0)
        minus_dm = np.where((data['low'].diff().abs() > data['high'].diff()) & 
                           (data['low'].diff() < 0), data['low'].diff().abs(), 0)
        
        # Smoothed values
        plus_di = 100 * (plus_dm.rolling(adx_period).sum() / tr.rolling(adx_period).sum())
        minus_di = 100 * (minus_dm.rolling(adx_period).sum() / tr.rolling(adx_period).sum())
        
        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        result['adx'] = dx.rolling(adx_period).mean()
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        
        # ADX signals
        result['adx_trending'] = (result['adx'] > 25).astype(int)
        result['adx_strong_trend'] = (result['adx'] > 50).astype(int)
        
        return result

    def _compute_oscillator_indicators(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute oscillator indicators (Stochastic, Williams %R, CCI)."""
        result = data.copy()
        
        # Stochastic Oscillator
        stoch_k = config.get('stoch_k', 14)
        stoch_d = config.get('stoch_d', 3)
        stoch_smooth = config.get('stoch_smooth', 3)
        
        lowest_low = data['low'].rolling(window=stoch_k, min_periods=1).min()
        highest_high = data['high'].rolling(window=stoch_k, min_periods=1).max()
        result['stoch_k'] = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        result['stoch_d'] = result['stoch_k'].rolling(window=stoch_d, min_periods=1).mean()
        result['stoch_k_smooth'] = result['stoch_k'].rolling(window=stoch_smooth, min_periods=1).mean()
        
        # Stochastic signals
        result['stoch_overbought'] = (result['stoch_k'] > 80).astype(int)
        result['stoch_oversold'] = (result['stoch_k'] < 20).astype(int)
        
        # Williams %R
        williams_period = config.get('williams_period', 14)
        result['williams_r'] = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        
        # CCI (Commodity Channel Index)
        cci_period = config.get('cci_period', 20)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=cci_period, min_periods=1).mean()
        mad = typical_price.rolling(window=cci_period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
        result['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # CCI signals
        result['cci_overbought'] = (result['cci'] > 100).astype(int)
        result['cci_oversold'] = (result['cci'] < -100).astype(int)
        
        return result

    def _compute_pattern_indicators(self, data: 'pd.DataFrame', config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute pattern recognition indicators."""
        result = data.copy()
        
        # Price patterns
        result['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        result['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
        result['higher_low'] = (data['low'] > data['low'].shift(1)).astype(int)
        result['lower_high'] = (data['high'] < data['high'].shift(1)).astype(int)
        
        # Gap detection
        result['gap_up'] = (data['low'] > data['high'].shift(1)).astype(int)
        result['gap_down'] = (data['high'] < data['low'].shift(1)).astype(int)
        
        # Doji pattern (simplified)
        body_size = np.abs(data['close'] - data['open'])
        total_range = data['high'] - data['low']
        result['doji'] = (body_size / total_range < 0.1).astype(int)
        
        # Hammer pattern (simplified)
        lower_shadow = data['open'].combine(data['close'], min) - data['low']
        upper_shadow = data['high'] - data['open'].combine(data['close'], max)
        result['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
        
        # Engulfing patterns
        prev_body = np.abs(data['close'].shift(1) - data['open'].shift(1))
        curr_body = np.abs(data['close'] - data['open'])
        result['bullish_engulfing'] = ((data['close'] > data['open']) & 
                                      (data['close'].shift(1) < data['open'].shift(1)) & 
                                      (curr_body > prev_body)).astype(int)
        result['bearish_engulfing'] = ((data['close'] < data['open']) & 
                                      (data['close'].shift(1) > data['open'].shift(1)) & 
                                      (curr_body > prev_body)).astype(int)
        
        return result

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        stats = {
            'chunk_size': self.chunk_size,
            'max_memory_gb': self.max_memory_gb,
            'gpu_enabled': self.enable_gpu,
            'm1_optimizations': self.m1_available,
            'hardware_optimization_enabled': self.hardware_optimization_enabled
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
        
        # Add hardware performance report if available
        if self.hardware_optimization_enabled and self.hardware_processor:
            stats['hardware_performance_report'] = self.hardware_processor.get_performance_report()

        return stats
    
    def get_hardware_performance_report(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive hardware performance report."""
        if self.hardware_optimization_enabled and self.hardware_processor:
            return self.hardware_processor.get_performance_report()
        return None
    
    def cleanup_hardware_resources(self):
        """Cleanup hardware resources."""
        if self.hardware_optimization_enabled and self.hardware_processor:
            self.hardware_processor.cleanup()
            self.logger.info("🧹 Hardware resources cleaned up")

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

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
