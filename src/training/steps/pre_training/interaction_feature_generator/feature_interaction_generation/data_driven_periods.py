"""
Data-Driven Period Selection for Cross-Timeframe Features (Refactored)

This module implements intelligent period selection based on data characteristics
rather than using hardcoded periods. It analyzes the data to determine optimal
periods for cross-timeframe feature generation.

Key Features:
- Analyzes data frequency and length
- Detects natural market cycles
- Optimizes periods for feature diversity
- Considers computational constraints
- Adapts to different timeframes (5m, 15m, 60m)
- VectorBT-optimized rolling operations
- Memory-efficient batch processing
- Parallel period analysis

Refactored Architecture:
- PeriodAnalyzer: Handles data analysis and pattern detection
- PeriodValidator: Handles filtering, ranking, and validation
- PeriodSelector: Coordinates the selection process
- PeriodAnalysisUtils: Common utilities to eliminate code duplication
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

# Import the new focused classes
from .period_analyzer import PeriodAnalyzer
from .period_validator import PeriodValidator
from .period_selector import PeriodSelector, PeriodAnalysisResult
from .period_analysis_utils import (
    PeriodAnalysisUtils, ValidationError, AnalysisError,
    performance_monitoring, safe_validate_and_execute
)

logger = logging.getLogger(__name__)


class DataDrivenPeriodSelector:
    """
    Selects optimal periods for cross-timeframe features based on data characteristics.
    
    This is the main API class that provides backward compatibility while using
    the new refactored architecture internally.
    
    Enhanced with VectorBT optimizations for improved performance and memory efficiency.
    """
    
    def __init__(self, 
                 min_period: int = 2,
                 max_period: int = 200,
                 max_periods: int = 8,
                 min_data_points: int = 100,
                 enable_vectorbt: bool = True,
                 enable_parallel: bool = True,
                 memory_efficient: bool = True,
                 chunk_size: int = 1000):
        """
        Initialize the period selector with VectorBT optimizations.
        
        Args:
            min_period: Minimum period to consider
            max_period: Maximum period to consider
            max_periods: Maximum number of periods to return
            min_data_points: Minimum data points required for analysis
            enable_vectorbt: Enable VectorBT optimizations
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            chunk_size: Size of data chunks for processing
        """
        self.min_period = min_period
        self.max_period = max_period
        self.max_periods = max_periods
        self.min_data_points = min_data_points
        self.enable_vectorbt = enable_vectorbt
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        
        # Initialize the internal period selector
        self._period_selector = PeriodSelector(
            min_period=min_period,
            max_period=max_period,
            max_periods=max_periods,
            min_data_points=min_data_points,
            enable_vectorbt=enable_vectorbt,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient,
            chunk_size=chunk_size
        )
        
        # Performance tracking (for backward compatibility)
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        tprint_info(f"🔧 Data-driven period selector initialized (refactored architecture)")
        tprint_info(f"📊 Period range: {min_period} - {max_period}")
        tprint_info(f"📊 Max periods: {max_periods}")
        tprint_info(f"🚀 VectorBT enabled: {enable_vectorbt}")
        tprint_info(f"⚡ Parallel processing: {enable_parallel}")
        tprint_info(f"💾 Memory efficient: {memory_efficient}")
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics to inform period selection using VectorBT optimizations.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Dictionary containing data characteristics
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        return self._period_selector.analyzer.analyze_data_characteristics(data)
    
    def select_optimal_periods(self, data: pd.DataFrame, 
                             target_timeframe: Optional[str] = None) -> PeriodAnalysisResult:
        """
        Select optimal periods for cross-timeframe features.
        
        Args:
            data: Input data
            target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
            
        Returns:
            PeriodAnalysisResult with optimal periods
        """
        return self._period_selector.select_optimal_periods(data, target_timeframe)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Get stats from internal selector
        internal_stats = self._period_selector.get_performance_stats()
        
        # Update our stats for backward compatibility
        self.performance_stats.update(internal_stats)
        
        return self.performance_stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._period_selector.reset_performance_stats()
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        tprint_success("✅ Performance statistics reset complete")
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        with performance_monitoring(operation_name):
            yield
    
    def optimize_for_large_datasets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for large dataset processing."""
        return self._period_selector.analyzer.optimize_for_large_datasets(data)
    
    def enable_cache(self, enabled: bool = True, max_size: int = 100):
        """Enable or disable caching."""
        self._period_selector.enable_cache(enabled, max_size)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._period_selector.get_cache_stats()
    
    # Backward compatibility methods
    def _detect_frequency(self, data: pd.DataFrame) -> str:
        """Detect the frequency of the data (backward compatibility)."""
        return PeriodAnalysisUtils.detect_frequency(data)
    
    def _get_timeframe_minutes(self, data: pd.DataFrame) -> int:
        """Get timeframe in minutes (backward compatibility)."""
        return PeriodAnalysisUtils.get_timeframe_minutes(data)
    
    def _find_pattern_periods(self, pattern: pd.Series) -> List[int]:
        """Find periods in a boolean pattern (backward compatibility)."""
        return PeriodAnalysisUtils.find_pattern_periods(pattern)
    
    def _calculate_confidence_score(self, periods: List[int], 
                                  characteristics: Dict[str, Any]) -> float:
        """Calculate confidence score (backward compatibility)."""
        return PeriodAnalysisUtils.calculate_confidence_score(periods, characteristics)


# Convenience functions (refactored to use new architecture)
def get_data_driven_periods(data: pd.DataFrame, 
                          target_timeframe: Optional[str] = None,
                          max_periods: int = 8,
                          enable_vectorbt: bool = True,
                          enable_parallel: bool = True,
                          memory_efficient: bool = True) -> List[int]:
    """
    Get data-driven periods for cross-timeframe features with VectorBT optimizations.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        enable_vectorbt: Enable VectorBT optimizations
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        
    Returns:
        List of optimal periods
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If analysis fails
    """
    tprint_info(f"🚀 Getting data-driven periods (data_shape: {data.shape}, target: {target_timeframe})")
    tprint_debug(f"📊 Configuration: max_periods={max_periods}, vectorbt={enable_vectorbt}, parallel={enable_parallel}, memory_efficient={memory_efficient}")
    
    # Fast fail for invalid inputs with comprehensive validation
    tprint_debug("🔍 Validating input parameters...")
    
    if not isinstance(data, pd.DataFrame):
        tprint_error("❌ Invalid input: expected pandas DataFrame")
        tprint_error(f"📊 Got: {type(data).__name__}")
        raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        tprint_error("❌ Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(max_periods, int) or max_periods <= 0:
        tprint_error("❌ Invalid max_periods: must be positive integer")
        tprint_error(f"📊 Got: {max_periods} (type: {type(max_periods).__name__})")
        raise ValueError("max_periods must be a positive integer")
    
    tprint_debug(f"✅ Input validation passed - data shape: {data.shape}, max_periods: {max_periods}")
    
    try:
        tprint_debug("🔧 Creating DataDrivenPeriodSelector instance...")
        selector = DataDrivenPeriodSelector(
            max_periods=max_periods,
            enable_vectorbt=enable_vectorbt,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient
        )
        
        tprint_debug("🔍 Selecting optimal periods...")
        result = selector.select_optimal_periods(data, target_timeframe)
        
        if result is None:
            tprint_error("❌ Selector returned None result")
            raise RuntimeError("Period selector returned None result")
        
        if not hasattr(result, 'optimal_periods') or result.optimal_periods is None:
            tprint_error("❌ Invalid result structure - missing optimal_periods")
            raise RuntimeError("Invalid result structure from period selector")
        
        tprint_success(f"✅ Data-driven periods retrieved: {result.optimal_periods}")
        tprint_debug(f"📊 Result confidence: {getattr(result, 'confidence_score', 'N/A')}")
        return result.optimal_periods
        
    except ValidationError as e:
        tprint_error(f"❌ Validation failed: {e}")
        tprint_error("📊 This indicates invalid input parameters")
        raise RuntimeError(f"Validation failed: {e}") from e
    except AnalysisError as e:
        tprint_error(f"❌ Analysis failed: {e}")
        tprint_error("📊 This indicates a problem with data analysis")
        raise RuntimeError(f"Analysis failed: {e}") from e
    except Exception as e:
        tprint_error(f"❌ Failed to get data-driven periods: {e}")
        tprint_error(f"📊 Error type: {type(e).__name__}")
        tprint_error("📊 This indicates an unexpected error")
        raise RuntimeError(f"Failed to get data-driven periods: {e}") from e


def get_data_driven_periods_with_stats(data: pd.DataFrame, 
                                     target_timeframe: Optional[str] = None,
                                     max_periods: int = 8,
                                     enable_vectorbt: bool = True,
                                     enable_parallel: bool = True,
                                     memory_efficient: bool = True) -> Tuple[List[int], Dict[str, Any]]:
    """
    Get data-driven periods with performance statistics.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        enable_vectorbt: Enable VectorBT optimizations
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        
    Returns:
        Tuple of (optimal periods, performance statistics)
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If analysis fails
    """
    tprint_info(f"🚀 Getting data-driven periods with stats (data_shape: {data.shape}, target: {target_timeframe})")
    tprint_debug(f"📊 Configuration: max_periods={max_periods}, vectorbt={enable_vectorbt}, parallel={enable_parallel}, memory_efficient={memory_efficient}")
    
    # Fast fail for invalid inputs
    if not isinstance(data, pd.DataFrame):
        tprint_error("❌ Invalid input: expected pandas DataFrame")
        raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        tprint_error("❌ Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(max_periods, int) or max_periods <= 0:
        tprint_error("❌ Invalid max_periods: must be positive integer")
        raise ValueError("max_periods must be a positive integer")
    
    try:
        selector = DataDrivenPeriodSelector(
            max_periods=max_periods,
            enable_vectorbt=enable_vectorbt,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient
        )
        result = selector.select_optimal_periods(data, target_timeframe)
        stats = selector.get_performance_stats()
        
        tprint_success(f"✅ Data-driven periods with stats retrieved: {result.optimal_periods}")
        tprint_debug(f"📊 Performance stats: {len(stats)} metrics collected")
        return result.optimal_periods, stats
        
    except Exception as e:
        tprint_error(f"❌ Failed to get data-driven periods with stats: {e}")
        raise RuntimeError(f"Failed to get data-driven periods with stats: {e}")


def benchmark_period_selector(data: pd.DataFrame, 
                            target_timeframe: Optional[str] = None,
                            max_periods: int = 8,
                            trials: int = 3) -> Dict[str, Any]:
    """
    Benchmark period selector performance across different configurations.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        trials: Number of trials to run for each configuration
        
    Returns:
        Benchmarking results
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If benchmarking fails
    """
    tprint_info(f"🚀 Starting period selector benchmark (data_shape: {data.shape}, trials: {trials})")
    tprint_debug(f"📊 Target timeframe: {target_timeframe}, max_periods: {max_periods}")
    
    # Fast fail for invalid inputs
    if not isinstance(data, pd.DataFrame):
        tprint_error("❌ Invalid input: expected pandas DataFrame")
        raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        tprint_error("❌ Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(trials, int) or trials <= 0:
        tprint_error("❌ Invalid trials: must be positive integer")
        raise ValueError("trials must be a positive integer")
    
    if not isinstance(max_periods, int) or max_periods <= 0:
        tprint_error("❌ Invalid max_periods: must be positive integer")
        raise ValueError("max_periods must be a positive integer")
    
    try:
        configurations = [
            {'enable_vectorbt': False, 'enable_parallel': False, 'memory_efficient': False, 'name': 'baseline'},
            {'enable_vectorbt': True, 'enable_parallel': False, 'memory_efficient': False, 'name': 'vectorbt_only'},
            {'enable_vectorbt': True, 'enable_parallel': True, 'memory_efficient': False, 'name': 'vectorbt_parallel'},
            {'enable_vectorbt': True, 'enable_parallel': True, 'memory_efficient': True, 'name': 'vectorbt_optimized'},
        ]
        
        results = {}
        
        for config in configurations:
            config_name = config.pop('name')
            tprint_info(f"🔄 Benchmarking configuration: {config_name}")
            tprint_debug(f"📊 Config: {config}")
            
            times = []
            
            for trial in range(trials):
                try:
                    tprint_debug(f"🔄 Trial {trial + 1}/{trials} for {config_name}")
                    selector = DataDrivenPeriodSelector(max_periods=max_periods, **config)
                    start_time = time.time()
                    result = selector.select_optimal_periods(data, target_timeframe)
                    execution_time = time.time() - start_time
                    times.append(execution_time)
                    tprint_debug(f"✅ Trial {trial + 1} completed in {execution_time:.3f}s")
                except Exception as e:
                    tprint_warning(f"⚠️ Configuration {config_name} trial {trial + 1} failed: {e}")
                    continue
            
            if times:
                results[config_name] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'trials_completed': len(times)
                }
                tprint_success(f"✅ {config_name}: {results[config_name]['avg_time']:.3f}s ± {results[config_name]['std_time']:.3f}s ({len(times)}/{trials} trials)")
            else:
                tprint_error(f"❌ {config_name}: All trials failed")
        
        tprint_success(f"✅ Benchmark complete: {len(results)} configurations tested")
        return results
        
    except Exception as e:
        tprint_error(f"❌ Benchmarking failed: {e}")
        raise RuntimeError(f"Benchmarking failed: {e}")


# Export the main classes and functions
__all__ = [
    'DataDrivenPeriodSelector',
    'PeriodAnalysisResult',
    'get_data_driven_periods',
    'get_data_driven_periods_with_stats',
    'benchmark_period_selector',
    'PeriodAnalyzer',
    'PeriodValidator',
    'PeriodSelector',
    'PeriodAnalysisUtils',
    'ValidationError',
    'AnalysisError'
]