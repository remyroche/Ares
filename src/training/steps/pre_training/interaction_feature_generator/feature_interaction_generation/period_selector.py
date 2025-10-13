"""
Period Selector for Data-Driven Period Selection

This module provides the main selection logic that coordinates between
analysis, validation, and final period selection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

from .period_analysis_utils import (
    PeriodAnalysisUtils, ValidationError, AnalysisError,
    performance_monitoring, safe_validate_and_execute
)
from .period_analyzer import PeriodAnalyzer
from .period_validator import PeriodValidator

logger = logging.getLogger(__name__)


@dataclass
class PeriodAnalysisResult:
    """Result of period analysis."""
    optimal_periods: List[int]
    period_categories: Dict[str, List[int]]
    analysis_metadata: Dict[str, Any]
    confidence_score: float


class PeriodSelector:
    """
    Main selector that coordinates period analysis, validation, and selection.
    
    This class orchestrates the entire period selection process by coordinating
    between the analyzer, validator, and providing the main API.
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
        Initialize the period selector.
        
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
        
        # Initialize components
        self.analyzer = PeriodAnalyzer(
            enable_vectorbt=enable_vectorbt,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient,
            chunk_size=chunk_size
        )
        
        self.validator = PeriodValidator(
            min_period=min_period,
            max_period=max_period,
            max_periods=max_periods,
            min_data_points=min_data_points
        )
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'analysis_operations': 0,
            'validation_operations': 0,
            'selection_operations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cache for computed results
        self._result_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 100
        
        tprint_info(f"🔧 Period selector initialized")
        tprint_info(f"📊 Period range: {min_period} - {max_period}")
        tprint_info(f"📊 Max periods: {max_periods}")
        tprint_info(f"📊 Min data points: {min_data_points}")
        tprint_info(f"🚀 VectorBT enabled: {enable_vectorbt}")
        tprint_info(f"⚡ Parallel processing: {enable_parallel}")
        tprint_info(f"💾 Memory efficient: {memory_efficient}")
    
    def select_optimal_periods(self, data: pd.DataFrame, 
                              target_timeframe: Optional[str] = None) -> PeriodAnalysisResult:
        """
        Select optimal periods for cross-timeframe features.
        
        Args:
            data: Input data
            target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
            
        Returns:
            PeriodAnalysisResult with optimal periods
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_dataframe(data, min_length=self.min_data_points, operation_name="period_selection")
        
        def _select_periods():
            tprint_info("🔍 Starting data-driven period selection...")
            
            # Check cache first
            if self._cache_enabled:
                cache_key = self._generate_cache_key('select_periods', data, target_timeframe)
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self.performance_stats['cache_hits'] += 1
                    tprint_success("✅ Cache hit - returning cached results")
                    return cached_result
                self.performance_stats['cache_misses'] += 1
            
            # Analyze data characteristics
            tprint_info("🔍 Analyzing data characteristics...")
            characteristics = self.analyzer.analyze_data_characteristics(data)
            self.performance_stats['analysis_operations'] += 1
            
            # Check if we have enough data
            if characteristics['data_length'] < self.min_data_points:
                raise AnalysisError(f"Insufficient data: {characteristics['data_length']} < {self.min_data_points} required")
            
            # Get base periods from timeframe
            base_periods = self.validator.get_base_periods_from_timeframe(
                characteristics.get('timeframe_minutes', 15),
                target_timeframe
            )
            
            # Analyze market cycles
            cycle_periods = self.analyzer.detect_market_cycles(data)
            self.performance_stats['analysis_operations'] += 1
            
            # Extract periods from characteristics
            volatility_periods = characteristics.get('volatility_clusters', [])
            volume_patterns = characteristics.get('volume_patterns', {})
            volume_periods = volume_patterns.get('spike_periods', []) if isinstance(volume_patterns, dict) else []
            
            # Combine all candidate periods
            all_candidate_periods = list(set(
                base_periods + cycle_periods + volatility_periods + volume_periods
            ))
            
            tprint_debug(f"📊 Combined {len(all_candidate_periods)} candidate periods")
            
            # Select optimal periods using validator
            optimal_periods = self.validator.select_optimal_periods(
                all_candidate_periods, data, characteristics
            )
            self.performance_stats['validation_operations'] += 1
            
            # Categorize periods
            period_categories = self.validator.categorize_periods(optimal_periods, characteristics)
            self.performance_stats['validation_operations'] += 1
            
            # Calculate confidence score
            confidence_score = self.validator.calculate_confidence_score(optimal_periods, characteristics)
            self.performance_stats['validation_operations'] += 1
            
            # Create result
            result = PeriodAnalysisResult(
                optimal_periods=optimal_periods,
                period_categories=period_categories,
                analysis_metadata=characteristics,
                confidence_score=confidence_score
            )
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
            
            self.performance_stats['selection_operations'] += 1
            self.performance_stats['total_operations'] += 1
            
            tprint_success(f"✅ Selected {len(optimal_periods)} optimal periods: {optimal_periods}")
            tprint_info(f"📊 Confidence score: {confidence_score:.2f}")
            
            return result
        
        return safe_validate_and_execute(
            _validate_inputs, _select_periods, "period_selection"
        )
    
    def get_data_driven_periods(self, data: pd.DataFrame, 
                               target_timeframe: Optional[str] = None) -> List[int]:
        """
        Get data-driven periods (convenience method).
        
        Args:
            data: Input data
            target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
            
        Returns:
            List of optimal periods
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        result = self.select_optimal_periods(data, target_timeframe)
        return result.optimal_periods
    
    def get_periods_with_metadata(self, data: pd.DataFrame, 
                                 target_timeframe: Optional[str] = None) -> Tuple[List[int], Dict[str, Any]]:
        """
        Get periods with analysis metadata.
        
        Args:
            data: Input data
            target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
            
        Returns:
            Tuple of (periods, metadata)
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        result = self.select_optimal_periods(data, target_timeframe)
        return result.optimal_periods, result.analysis_metadata
    
    def validate_period_quality(self, data: pd.DataFrame, 
                               target_timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate the quality of selected periods.
        
        Args:
            data: Input data
            target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
            
        Returns:
            Dictionary with quality metrics
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        result = self.select_optimal_periods(data, target_timeframe)
        
        return self.validator.validate_period_quality(
            result.optimal_periods, data, result.analysis_metadata
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add analyzer stats
        analyzer_stats = self.analyzer.get_performance_stats()
        stats.update({f"analyzer_{k}": v for k, v in analyzer_stats.items()})
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['analysis_usage_rate'] = stats['analysis_operations'] / stats['total_operations']
            stats['validation_usage_rate'] = stats['validation_operations'] / stats['total_operations']
            stats['selection_usage_rate'] = stats['selection_operations'] / stats['total_operations']
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['average_operation_time'] = 0
            stats['analysis_usage_rate'] = 0
            stats['validation_usage_rate'] = 0
            stats['selection_usage_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'analysis_operations': 0,
            'validation_operations': 0,
            'selection_operations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Reset component stats
        self.analyzer.reset_performance_stats()
        
        # Clear cache
        cache_size = len(self._result_cache)
        self._result_cache.clear()
        tprint_debug(f"🗑️ Cleared cache ({cache_size} entries removed)")
    
    def _generate_cache_key(self, operation: str, data: pd.DataFrame, target_timeframe: Optional[str] = None) -> str:
        """Generate cache key for operation."""
        import hashlib
        
        # Create hash of data characteristics and operation
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        timeframe_hash = hashlib.md5(str(target_timeframe).encode()).hexdigest()[:4]
        cache_key = f"{operation}_{data_hash}_{timeframe_hash}"
        
        return cache_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[PeriodAnalysisResult]:
        """Get result from cache."""
        if not self._cache_enabled:
            return None
        
        if cache_key in self._result_cache:
            tprint_debug(f"✅ Cache hit for key: {cache_key}")
            return self._result_cache[cache_key]
        else:
            tprint_debug(f"❌ Cache miss for key: {cache_key}")
            return None
    
    def _put_in_cache(self, cache_key: str, result: PeriodAnalysisResult):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        # Limit cache size
        if len(self._result_cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]
            tprint_debug(f"🗑️ Removed oldest cache entry: {oldest_key}")
        
        self._result_cache[cache_key] = result
        tprint_debug(f"✅ Stored result in cache (size: {len(self._result_cache)}/{self._max_cache_size})")
    
    def enable_cache(self, enabled: bool = True, max_size: int = 100):
        """Enable or disable caching."""
        self._cache_enabled = enabled
        self._max_cache_size = max_size
        
        if not enabled:
            self._result_cache.clear()
            tprint_info("🗑️ Cache disabled and cleared")
        else:
            tprint_info(f"✅ Cache enabled (max size: {max_size})")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_enabled': self._cache_enabled,
            'cache_size': len(self._result_cache),
            'max_cache_size': self._max_cache_size,
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_misses': self.performance_stats['cache_misses'],
            'cache_hit_rate': self.performance_stats.get('cache_hit_rate', 0)
        }