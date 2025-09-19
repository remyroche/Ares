"""
Memory Optimizer for Backtesting Operations

This module provides specialized memory optimization for backtesting operations,
integrating with the hardware optimization tools and matrix operations utilities.

Key Features:
- DataFrame memory optimization and cleanup
- Automatic garbage collection during backtesting
- Memory usage monitoring and alerting
- Context managers for memory-intensive operations
- Integration with M1 hardware optimizations
"""

import logging
import gc
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import weakref
from pathlib import Path

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import hardware optimization tools
try:
    from src.utils.hardware.advanced_memory_optimizer import AdvancedMemoryOptimizer
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel, HardwareConfig
    from src.utils.hardware.memory_optimization import MemoryOptimizer
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    from src.utils.matrix_operations.batch_operations import BatchProcessor
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    timestamp: datetime
    process_memory_mb: float
    process_memory_percent: float
    available_memory_mb: float
    dataframe_count: int
    total_dataframe_memory_mb: float
    gc_collections: int


class BacktestingMemoryOptimizer:
    """Memory optimizer specifically designed for backtesting operations."""
    
    def __init__(self, memory_limit_mb: float = 2000.0, enable_monitoring: bool = True):
        """Initialize backtesting memory optimizer.
        
        Args:
            memory_limit_mb: Memory limit in megabytes
            enable_monitoring: Enable memory monitoring
        """
        self.memory_limit_mb = memory_limit_mb
        self.enable_monitoring = enable_monitoring
        self.logger = logger.getChild('BacktestingMemoryOptimizer')
        
        # Initialize hardware optimizers using unified hardware manager
        self.unified_hardware_manager = None
        self.m1_memory_optimizer = None
        self.advanced_memory_optimizer = None
        self.matrix_ops = None
        self.batch_processor = None
        
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                # Initialize unified hardware manager
                self.unified_hardware_manager = UnifiedHardwareManager()
                
                # Configure for backtesting workload
                hardware_config = HardwareConfig(
                    memory_optimization_level=OptimizationLevel.BALANCED,
                    memory_limit_gb=memory_limit_mb / 1024,
                    enable_memory_pooling=True,
                    enable_predictive_allocation=True,
                    enable_compression=True
                )
                
                # Initialize specific optimizers
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.advanced_memory_optimizer = AdvancedMemoryOptimizer()
                self.matrix_ops = UnifiedMatrixOperations()
                self.batch_processor = BatchProcessor()
                
                self.logger.info("✅ Unified hardware optimization enabled")
            except Exception as e:
                self.logger.warning(f"Hardware optimization not available: {e}")
        
        # Track DataFrames for cleanup
        self.tracked_dataframes: List[weakref.ref] = []
        self.memory_stats_history: List[MemoryStats] = []
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        
        if enable_monitoring:
            self.start_monitoring()
        
        self.logger.info(f"✅ BacktestingMemoryOptimizer initialized (limit: {memory_limit_mb:.1f}MB)")
    
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, monitoring disabled")
            return
        
        self.monitoring_active = True
        self.logger.info("📊 Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring_active = False
        self.logger.info("📊 Memory monitoring stopped")
    
    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        stats = MemoryStats(
            timestamp=datetime.now(),
            process_memory_mb=0.0,
            process_memory_percent=0.0,
            available_memory_mb=0.0,
            dataframe_count=0,
            total_dataframe_memory_mb=0.0,
            gc_collections=sum(gc.get_stats())
        )
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                virtual_memory = psutil.virtual_memory()
                
                stats.process_memory_mb = memory_info.rss / 1024 / 1024
                stats.process_memory_percent = process.memory_percent()
                stats.available_memory_mb = virtual_memory.available / 1024 / 1024
            except Exception as e:
                self.logger.warning(f"Could not get system memory stats: {e}")
        
        # Count active DataFrames
        active_dfs = []
        for ref in self.tracked_dataframes:
            df = ref()
            if df is not None:
                active_dfs.append(df)
                try:
                    stats.total_dataframe_memory_mb += df.memory_usage(deep=True).sum() / 1024 / 1024
                except Exception:
                    pass
        
        stats.dataframe_count = len(active_dfs)
        self.tracked_dataframes = [weakref.ref(df) for df in active_dfs]
        
        return stats
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        stats = self.get_current_memory_stats()
        
        if stats.process_memory_mb > self.memory_limit_mb:
            self.logger.warning(
                f"⚠️ Memory usage exceeds limit: {stats.process_memory_mb:.1f}MB > {self.memory_limit_mb:.1f}MB"
            )
            return False
        
        return True
    
    def optimize_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Optimize DataFrame memory usage using unified hardware tools.
        
        Args:
            df: DataFrame to optimize
            inplace: Whether to modify DataFrame in place
            
        Returns:
            Optimized DataFrame
        """
        if not PANDAS_AVAILABLE:
            return df
        
        target_df = df if inplace else df.copy()
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        try:
            # Use unified hardware manager for optimization if available
            if self.unified_hardware_manager:
                # Configure for data processing optimization
                optimization_config = {
                    'workload_type': WorkloadType.DATA_PROCESSING,
                    'optimization_level': OptimizationLevel.BALANCED
                }
                
                with self.unified_hardware_manager.optimize_for_workload(**optimization_config):
                    if self.matrix_ops:
                        target_df = self.matrix_ops.optimize_dataframe_memory(target_df)
                    else:
                        target_df = self._basic_dataframe_optimization(target_df)
                        
            elif self.matrix_ops:
                # Use matrix operations for optimization
                target_df = self.matrix_ops.optimize_dataframe_memory(target_df)
            else:
                # Fallback optimization
                target_df = self._basic_dataframe_optimization(target_df)
            
            optimized_memory = target_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            if optimized_memory < original_memory:
                reduction_percent = ((original_memory - optimized_memory) / original_memory * 100)
                self.logger.debug(
                    f"🧠 DataFrame optimized: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                    f"({reduction_percent:.1f}% reduction)"
                )
            
            # Track the DataFrame
            if not inplace:
                self.track_dataframe(target_df)
            
            return target_df
            
        except Exception as e:
            self.logger.error(f"❌ DataFrame optimization failed: {e}")
            return target_df
    
    def _basic_dataframe_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame memory optimization."""
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max <= 255:
                    df[col] = df[col].astype('uint8')
                elif col_max <= 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max <= 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def track_dataframe(self, df: pd.DataFrame) -> None:
        """Track DataFrame for memory management."""
        self.tracked_dataframes.append(weakref.ref(df))
    
    def cleanup_dataframes(self) -> int:
        """Clean up tracked DataFrames and return count of cleaned items."""
        cleaned_count = 0
        active_refs = []
        
        for ref in self.tracked_dataframes:
            df = ref()
            if df is not None:
                active_refs.append(ref)
            else:
                cleaned_count += 1
        
        self.tracked_dataframes = active_refs
        return cleaned_count
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force comprehensive memory cleanup using unified hardware manager."""
        start_stats = self.get_current_memory_stats()
        
        # Clean up tracked DataFrames
        cleaned_dfs = self.cleanup_dataframes()
        
        # Use unified hardware manager for comprehensive cleanup
        if self.unified_hardware_manager:
            try:
                # Force comprehensive hardware cleanup
                hardware_cleanup = self.unified_hardware_manager.force_cleanup()
                self.logger.debug(f"Hardware cleanup result: {hardware_cleanup}")
            except Exception as e:
                self.logger.warning(f"Hardware cleanup failed: {e}")
        
        # Force memory optimizer cleanup
        if self.m1_memory_optimizer:
            try:
                self.m1_memory_optimizer.force_cleanup()
            except Exception as e:
                self.logger.warning(f"M1 memory cleanup failed: {e}")
        
        # Force advanced memory optimizer cleanup
        if self.advanced_memory_optimizer:
            try:
                self.advanced_memory_optimizer.cleanup()
            except Exception as e:
                self.logger.warning(f"Advanced memory cleanup failed: {e}")
        
        # Multiple garbage collection passes
        gc_collections = 0
        for i in range(3):
            collected = gc.collect()
            gc_collections += collected
            if collected == 0:
                break
        
        end_stats = self.get_current_memory_stats()
        
        cleanup_result = {
            'cleaned_dataframes': cleaned_dfs,
            'memory_before_mb': start_stats.process_memory_mb,
            'memory_after_mb': end_stats.process_memory_mb,
            'memory_freed_mb': start_stats.process_memory_mb - end_stats.process_memory_mb,
            'gc_collections': gc_collections,
            'hardware_optimization_used': self.unified_hardware_manager is not None
        }
        
        self.logger.info(f"🧹 Force cleanup completed:")
        self.logger.info(f"   📊 DataFrames cleaned: {cleaned_dfs}")
        self.logger.info(f"   🧠 Memory freed: {cleanup_result['memory_freed_mb']:.1f}MB")
        self.logger.info(f"   🗑️ GC collections: {cleanup_result['gc_collections']}")
        self.logger.info(f"   ⚡ Hardware optimization: {cleanup_result['hardware_optimization_used']}")
        
        return cleanup_result
    
    @contextmanager
    def memory_managed_operation(self, operation_name: str = "operation"):
        """Context manager for memory-managed operations using unified hardware manager."""
        start_stats = self.get_current_memory_stats()
        self.logger.debug(f"🚀 Starting memory-managed {operation_name}")
        
        try:
            # Use unified hardware manager for comprehensive optimization
            if self.unified_hardware_manager:
                operation_config = {
                    'workload_type': WorkloadType.DATA_PROCESSING,
                    'optimization_level': OptimizationLevel.BALANCED,
                    'memory_limit_gb': self.memory_limit_mb / 1024
                }
                
                with self.unified_hardware_manager.optimize_for_workload(**operation_config):
                    yield
                    
            elif self.m1_memory_optimizer:
                # Fallback to M1 memory optimization
                with self.m1_memory_optimizer.optimization_context():
                    yield
            else:
                # Basic operation without hardware optimization
                yield
                
        finally:
            # Cleanup after operation
            self.cleanup_dataframes()
            
            # Check memory usage
            if not self.check_memory_usage():
                self.logger.warning(f"⚠️ Memory limit exceeded after {operation_name}")
                self.force_cleanup()
            
            end_stats = self.get_current_memory_stats()
            self.logger.debug(
                f"✅ Completed {operation_name} "
                f"(memory: {start_stats.process_memory_mb:.1f}MB → {end_stats.process_memory_mb:.1f}MB)"
            )
    
    def optimize_backtesting_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Optimize multiple DataFrames for backtesting operations.
        
        Args:
            dataframes: Dictionary of DataFrames to optimize
            
        Returns:
            Dictionary of optimized DataFrames
        """
        optimized = {}
        total_original_memory = 0.0
        total_optimized_memory = 0.0
        
        for name, df in dataframes.items():
            if df.empty:
                optimized[name] = df
                continue
            
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            total_original_memory += original_memory
            
            optimized_df = self.optimize_dataframe(df, inplace=False)
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
            total_optimized_memory += optimized_memory
            
            optimized[name] = optimized_df
            
            self.logger.debug(f"📊 {name}: {original_memory:.1f}MB → {optimized_memory:.1f}MB")
        
        if total_original_memory > 0:
            reduction_percent = ((total_original_memory - total_optimized_memory) / total_original_memory) * 100
            self.logger.info(
                f"🧠 Total optimization: {total_original_memory:.1f}MB → {total_optimized_memory:.1f}MB "
                f"({reduction_percent:.1f}% reduction)"
            )
        
        return optimized
    
    def process_large_dataframe_in_chunks(
        self,
        df: pd.DataFrame,
        chunk_size: int,
        processing_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """Process large DataFrame in memory-optimized chunks using unified hardware tools.
        
        Args:
            df: Large DataFrame to process
            chunk_size: Size of each chunk
            processing_func: Function to apply to each chunk
            **kwargs: Additional arguments for processing function
            
        Returns:
            Processed DataFrame
        """
        # Use unified hardware manager for optimal chunk processing
        if self.unified_hardware_manager:
            try:
                # Configure for data processing workload
                processing_config = {
                    'workload_type': WorkloadType.DATA_PROCESSING,
                    'optimization_level': OptimizationLevel.AGGRESSIVE,
                    'memory_limit_gb': self.memory_limit_mb / 1024
                }
                
                with self.unified_hardware_manager.optimize_for_workload(**processing_config):
                    if self.batch_processor:
                        return self.batch_processor.process_in_batches(
                            df, batch_size=chunk_size, operation=processing_func, **kwargs
                        )
                    else:
                        return self._fallback_chunk_processing(df, chunk_size, processing_func, **kwargs)
                        
            except Exception as e:
                self.logger.warning(f"Unified hardware processing failed, using fallback: {e}")
                return self._fallback_chunk_processing(df, chunk_size, processing_func, **kwargs)
        
        elif self.batch_processor:
            try:
                return self.batch_processor.process_in_batches(
                    df, batch_size=chunk_size, operation=processing_func, **kwargs
                )
            except Exception as e:
                self.logger.warning(f"Batch processor failed, using fallback: {e}")
                return self._fallback_chunk_processing(df, chunk_size, processing_func, **kwargs)
        
        else:
            return self._fallback_chunk_processing(df, chunk_size, processing_func, **kwargs)
    
    def _fallback_chunk_processing(
        self,
        df: pd.DataFrame,
        chunk_size: int,
        processing_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """Fallback chunk processing implementation."""
        chunks = []
        total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
        
        self.logger.info(f"🔄 Processing {len(df):,} rows in {total_chunks} chunks of {chunk_size:,}")
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            
            with self.memory_managed_operation(f"chunk_{i//chunk_size + 1}"):
                processed_chunk = processing_func(chunk, **kwargs)
                chunks.append(processed_chunk)
                
                # Clean up chunk
                del chunk
                gc.collect()
        
        # Concatenate results
        with self.memory_managed_operation("concatenation"):
            result = pd.concat(chunks, ignore_index=True)
            
            # Clean up chunks
            for chunk in chunks:
                del chunk
            del chunks
            gc.collect()
        
        return result
    
    def optimize_equity_curve_calculation(
        self,
        trades: pd.DataFrame,
        initial_capital: float
    ) -> pd.DataFrame:
        """Memory-optimized equity curve calculation using unified hardware tools.
        
        Args:
            trades: DataFrame with trade information
            initial_capital: Initial portfolio value
            
        Returns:
            Equity curve DataFrame
        """
        if trades.empty:
            return pd.DataFrame()
        
        # Use unified hardware manager for optimal vectorized calculation
        if self.unified_hardware_manager:
            calculation_config = {
                'workload_type': WorkloadType.DATA_PROCESSING,
                'optimization_level': OptimizationLevel.AGGRESSIVE
            }
            
            with self.unified_hardware_manager.optimize_for_workload(**calculation_config):
                return self._calculate_equity_curve_optimized(trades, initial_capital)
        else:
            return self._calculate_equity_curve_optimized(trades, initial_capital)
    
    def _calculate_equity_curve_optimized(self, trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """Optimized equity curve calculation implementation."""
        with self.memory_managed_operation("equity_curve_calculation"):
            # Use vectorized operations for memory efficiency
            if 'pnl' in trades.columns:
                # Use matrix operations for efficient cumulative sum if available
                if self.matrix_ops:
                    try:
                        cumulative_pnl = self.matrix_ops.cumulative_sum(trades['pnl'].values)
                        cumulative_pnl = pd.Series(cumulative_pnl, index=trades.index)
                    except Exception as e:
                        self.logger.debug(f"Matrix ops cumsum failed, using pandas: {e}")
                        cumulative_pnl = trades['pnl'].cumsum()
                else:
                    cumulative_pnl = trades['pnl'].cumsum()
                
                equity_curve = pd.DataFrame({
                    'timestamp': trades['timestamp'] if 'timestamp' in trades.columns else trades.index,
                    'equity': initial_capital + cumulative_pnl,
                    'pnl': trades['pnl'],
                    'cumulative_pnl': cumulative_pnl
                })
            else:
                # Fallback if no PnL column
                equity_curve = pd.DataFrame({
                    'timestamp': trades['timestamp'] if 'timestamp' in trades.columns else trades.index,
                    'equity': [initial_capital] * len(trades)
                })
            
            # Optimize the resulting DataFrame
            equity_curve = self.optimize_dataframe(equity_curve)
            
            return equity_curve
    
    def calculate_rolling_metrics_optimized(
        self,
        data: pd.DataFrame,
        window: int,
        metrics: List[str]
    ) -> pd.DataFrame:
        """Calculate rolling metrics with unified hardware optimization.
        
        Args:
            data: Input DataFrame
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if data.empty or 'close' not in data.columns:
            return pd.DataFrame()
        
        # Use unified hardware manager for optimal rolling calculations
        if self.unified_hardware_manager:
            metrics_config = {
                'workload_type': WorkloadType.DATA_PROCESSING,
                'optimization_level': OptimizationLevel.BALANCED
            }
            
            with self.unified_hardware_manager.optimize_for_workload(**metrics_config):
                return self._calculate_rolling_metrics_impl(data, window, metrics)
        else:
            return self._calculate_rolling_metrics_impl(data, window, metrics)
    
    def _calculate_rolling_metrics_impl(self, data: pd.DataFrame, window: int, metrics: List[str]) -> pd.DataFrame:
        """Implementation of rolling metrics calculation."""
        with self.memory_managed_operation("rolling_metrics"):
            result_data = {'timestamp': data.index}
            
            # Pre-calculate returns once for efficiency
            returns = data['close'].pct_change()
            
            # Use matrix operations for rolling calculations if available
            if self.matrix_ops:
                try:
                    # Calculate metrics efficiently using matrix operations
                    if 'volatility' in metrics:
                        volatility_values = self.matrix_ops.rolling_std(returns.values, window) * np.sqrt(252)
                        result_data['volatility'] = pd.Series(volatility_values, index=data.index)
                    
                    if 'sharpe_ratio' in metrics:
                        if 'volatility' not in result_data:
                            volatility_values = self.matrix_ops.rolling_std(returns.values, window) * np.sqrt(252)
                            volatility = pd.Series(volatility_values, index=data.index)
                        else:
                            volatility = result_data['volatility']
                        
                        mean_return_values = self.matrix_ops.rolling_mean(returns.values, window) * 252
                        mean_return = pd.Series(mean_return_values, index=data.index)
                        result_data['sharpe_ratio'] = mean_return / volatility
                    
                    if 'max_drawdown' in metrics:
                        rolling_max_values = self.matrix_ops.rolling_max(data['close'].values, window)
                        rolling_max = pd.Series(rolling_max_values, index=data.index)
                        drawdown = (data['close'] - rolling_max) / rolling_max
                        drawdown_min_values = self.matrix_ops.rolling_min(drawdown.values, window)
                        result_data['max_drawdown'] = pd.Series(drawdown_min_values, index=data.index)
                        
                except Exception as e:
                    self.logger.debug(f"Matrix ops rolling calculation failed, using pandas: {e}")
                    return self._calculate_rolling_metrics_pandas(data, window, metrics, returns)
            else:
                return self._calculate_rolling_metrics_pandas(data, window, metrics, returns)
            
            # Create optimized DataFrame
            result_df = pd.DataFrame(result_data)
            result_df = self.optimize_dataframe(result_df)
            
            return result_df
    
    def _calculate_rolling_metrics_pandas(self, data: pd.DataFrame, window: int, metrics: List[str], returns: pd.Series) -> pd.DataFrame:
        """Fallback pandas implementation for rolling metrics."""
        result_data = {'timestamp': data.index}
        
        # Calculate metrics efficiently using pandas
        if 'volatility' in metrics:
            result_data['volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        if 'sharpe_ratio' in metrics:
            if 'volatility' not in result_data:
                volatility = returns.rolling(window).std() * np.sqrt(252)
            else:
                volatility = result_data['volatility']
            
            mean_return = returns.rolling(window).mean() * 252
            result_data['sharpe_ratio'] = mean_return / volatility
        
        if 'max_drawdown' in metrics:
            rolling_max = data['close'].rolling(window).max()
            drawdown = (data['close'] - rolling_max) / rolling_max
            result_data['max_drawdown'] = drawdown.rolling(window).min()
        
        # Create optimized DataFrame
        result_df = pd.DataFrame(result_data)
        result_df = self.optimize_dataframe(result_df)
        
        return result_df
    
    @contextmanager
    def backtesting_session(self, session_name: str = "backtesting"):
        """Context manager for entire backtesting session using unified hardware manager."""
        self.logger.info(f"🚀 Starting {session_name} session with unified hardware optimization")
        start_stats = self.get_current_memory_stats()
        
        try:
            if self.unified_hardware_manager:
                # Use unified hardware manager for comprehensive optimization
                workload_config = {
                    'workload_type': WorkloadType.BACKTESTING,
                    'optimization_level': OptimizationLevel.BALANCED,
                    'memory_limit_gb': self.memory_limit_mb / 1024,
                    'enable_monitoring': self.enable_monitoring
                }
                
                with self.unified_hardware_manager.optimize_for_workload(**workload_config):
                    yield self
                    
            elif self.advanced_memory_optimizer:
                # Fallback to advanced memory optimizer
                with self.advanced_memory_optimizer.optimization_context():
                    yield self
            else:
                # Basic session without hardware optimization
                yield self
                
        finally:
            # Comprehensive cleanup
            cleanup_result = self.force_cleanup()
            end_stats = self.get_current_memory_stats()
            
            self.logger.info(f"✅ {session_name} session completed")
            self.logger.info(f"   🧠 Memory at start: {start_stats.process_memory_mb:.1f}MB")
            self.logger.info(f"   🧠 Memory at end: {end_stats.process_memory_mb:.1f}MB")
            self.logger.info(f"   🧹 Memory freed: {cleanup_result['memory_freed_mb']:.1f}MB")
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        stats = self.get_current_memory_stats()
        
        if stats.process_memory_mb > self.memory_limit_mb * 0.8:
            recommendations.append("Memory usage is high - consider reducing chunk sizes")
        
        if stats.total_dataframe_memory_mb > stats.process_memory_mb * 0.5:
            recommendations.append("DataFrames consume significant memory - enable aggressive optimization")
        
        if stats.dataframe_count > 50:
            recommendations.append("Many DataFrames in memory - consider more frequent cleanup")
        
        if PSUTIL_AVAILABLE and stats.available_memory_mb < 1000:
            recommendations.append("System memory is low - reduce memory limits")
        
        return recommendations


# Global instance
_backtesting_memory_optimizer = None

def get_backtesting_memory_optimizer(
    memory_limit_mb: float = 2000.0,
    enable_monitoring: bool = True
) -> BacktestingMemoryOptimizer:
    """Get global backtesting memory optimizer instance."""
    global _backtesting_memory_optimizer
    if _backtesting_memory_optimizer is None:
        _backtesting_memory_optimizer = BacktestingMemoryOptimizer(
            memory_limit_mb=memory_limit_mb,
            enable_monitoring=enable_monitoring
        )
    return _backtesting_memory_optimizer


# Convenience functions for common operations
def optimize_backtesting_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Optimize multiple DataFrames for backtesting."""
    optimizer = get_backtesting_memory_optimizer()
    return optimizer.optimize_backtesting_dataframes(data)


def cleanup_backtesting_memory() -> Dict[str, Any]:
    """Force cleanup of backtesting memory."""
    optimizer = get_backtesting_memory_optimizer()
    return optimizer.force_cleanup()


@contextmanager
def memory_managed_backtesting(session_name: str = "backtesting"):
    """Context manager for memory-managed backtesting operations."""
    optimizer = get_backtesting_memory_optimizer()
    with optimizer.backtesting_session(session_name):
        yield optimizer