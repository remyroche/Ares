"""
import warnings
Optimized Cross Timeframe Analysis Module

This module provides highly optimized cross timeframe analysis leveraging:
- M1 hardware optimizations (CPU/GPU/Memory)
- Advanced feature selection from step08 utilities
- Enhanced data quality validation
- Parallel processing and caching
- Memory-efficient operations

Key Features:
- M1-optimized processing with
- Advanced feature selection with regime awareness
- Comprehensive data quality validation
- Intelligent caching and memory management
- Parallel processing with optimized thread pools
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Core imports
from src.utils.logger import system_logger
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range,
    safe_divide, safe_log, safe_sqrt, safe_power,
    MathValidationError
)

# Hardware optimization imports
HARDWARE_OPTIMIZATIONS_AVAILABLE = False
try:
                HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False

# Advanced feature selection imports
# NOTE: step08_* modules no longer exist. Use src.feature_selection instead.
FEATURE_SELECTION_AVAILABLE = False
try:
    # These imports will fail - kept for backward compatibility
    from src.feature_selection.step08_unified_final import (
        Step08Unified, FinancialMetrics, RiskMetrics, RegimeBalanceMetrics,
        FeatureSelectionValidation, Step08Results
    )
    from src.feature_selection.step08_advanced_feature_selection import (
        Step08AdvancedFeatureSelection
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    # Create dummy classes for fallback
    class Step08Unified:
        pass
    class FinancialMetrics:
        pass
    class RiskMetrics:
        pass
    class RegimeBalanceMetrics:
        pass
    class FeatureSelectionValidation:
        pass
    class Step08Results:
        pass
    class Step08AdvancedFeatureSelection:
        pass
    FEATURE_SELECTION_AVAILABLE = False

# Data quality and processing utilities
try:
    from src.utils.data import DataFrameValidator, DataFrameCleaner, DataFrameTransformer
    from src.utils.parquet_utils import ParquetUtils
    DATA_VALIDATION_AVAILABLE = True
except ImportError:
    # Create dummy classes for fallback
    class DataFrameValidator:
        pass
    class DataFrameCleaner:
        pass
    class DataFrameTransformer:
        pass
    class ParquetUtils:
        pass
    DATA_VALIDATION_AVAILABLE = False
    from src.utils.serialization_utils import JSONSerializer, ParquetSerializer
    from src.utils.caching import IntelligentCache
    UTILITIES_AVAILABLE = True
except ImportError:
    UTILITIES_AVAILABLE = False

# ML Commons
try:
    from src.utils.ml_common import (
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
        DataQualityUtilities, FeatureSelectionFramework,
        CrossValidationUtilities
    )
    ML_COMMONS_AVAILABLE = True
except ImportError:
    ML_COMMONS_AVAILABLE = False

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

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

except ImportError:

    cp = None

logger = system_logger.getChild('OptimizedCrossTimeframeAnalysis')

@dataclass
class OptimizedCrossTimeframeConfig:
    """Configuration for optimized cross timeframe analysis."""
    # Timeframe configuration
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '30m'])
    base_timeframe: str = '1m'

    # Feature engineering
    interaction_features: List[str] = field(default_factory=lambda: [
        'correlation', 'momentum', 'volatility', 'volume', 'microstructure',
        'order_flow', 'momentum_divergence', 'volatility_spillover'
    ])
    lookback_periods: List[int] = field(default_factory=lambda: [3, 5, 10, 15, 20, 30])

    # Analysis parameters
    correlation_threshold: float = 0.6
    min_observations: int = 50
    max_correlations: int = 30

    # Hardware optimization
    enable_m1_optimizations: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    max_workers: int = 4

    # Advanced feature selection
    enable_advanced_feature_selection: bool = True
    feature_selection_method: str = 'mutual_info'
    redundancy_threshold: float = 0.8

    # Data quality
    enable_data_quality_validation: bool = True
    quality_thresholds: Dict[str, Any] = field(default_factory=dict)

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

@dataclass
class OptimizedCrossTimeframeResult:
    """Result of optimized cross timeframe analysis."""
    cross_timeframe_features: pd.DataFrame
    selected_features: Dict[str, List[str]]
    interaction_metrics: Dict[str, Any]
    timeframe_correlations: Dict[str, Any]
    feature_importance: Dict[str, Any]
    financial_metrics: Optional[FinancialMetrics] = None
    risk_metrics: Optional[RiskMetrics] = None
    quality_report: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

class OptimizedCrossTimeframeAnalysis:
    """
    Optimized Cross Timeframe Analysis with M1 hardware acceleration and advanced feature selection.
    """

    def __init__(self, config: Optional[OptimizedCrossTimeframeConfig] = None):
        """Initialize optimized cross timeframe analysis."""
        self.config = config or OptimizedCrossTimeframeConfig()
        self.logger = logger.getChild('OptimizedCrossTimeframeAnalysis')

        # Initialize hardware optimizers
        self._init_hardware_optimizers()

        # Initialize feature selection
        self._init_feature_selection()

        # Initialize utilities
        self._init_utilities()

        # Initialize caching
        self._init_caching()

        self.logger.info("✅ Optimized Cross Timeframe Analysis initialized")

    def _init_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        if HARDWARE_OPTIMIZATIONS_AVAILABLE and self.config.enable_m1_optimizations:
            try:
                self.memory_optimizer = get_integrated_hardware_manager()
                self.cpu_optimizer = get_comprehensive_optimizer()
                self.gpu_manager = get_integrated_hardware_manager()

                self.logger.info("✅ Hardware optimizers initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.gpu_manager = None
        else:
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.gpu_manager = None

    def _init_feature_selection(self):
        """Initialize advanced feature selection components."""
        if FEATURE_SELECTION_AVAILABLE and self.config.enable_advanced_feature_selection:
            try:
                # Initialize advanced feature selection
                feature_config = {
                    'use_m1_optimizations': self.config.enable_m1_optimizations,
                    'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                    'memory_limit_gb': self.config.memory_limit_gb,
                    'max_workers': self.config.max_workers,
                    'feature_selection_method': self.config.feature_selection_method,
                    'redundancy_threshold': self.config.redundancy_threshold
                }

                self.feature_selector = Step08AdvancedFeatureSelection(feature_config)
                self.logger.info("✅ Advanced feature selection initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Advanced feature selection initialization failed: {e}")
                self.feature_selector = None
        else:
            self.feature_selector = None

    def _init_utilities(self):
        """Initialize utility components."""
        if UTILITIES_AVAILABLE:
            try:
                self.data_validator = DataFrameValidator()
                self.data_cleaner = DataFrameCleaner()
                self.data_transformer = DataFrameTransformer()
                self.parquet_utils = ParquetUtils()
                self.json_serializer = JSONSerializer()
                self.parquet_serializer = ParquetSerializer()

                self.logger.info("✅ Utility components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Utility initialization failed: {e}")
                self.data_validator = None
                self.data_cleaner = None
                self.data_transformer = None
                self.parquet_utils = None
                self.json_serializer = None
                self.parquet_serializer = None
        else:
            self.data_validator = None
            self.data_cleaner = None
            self.data_transformer = None
            self.parquet_utils = None
            self.json_serializer = None
            self.parquet_serializer = None

    def _init_caching(self):
        """Initialize caching system."""
        if UTILITIES_AVAILABLE and self.config.enable_caching:
            try:
                self.cache = IntelligentCache(ttl_seconds=self.config.cache_ttl_seconds)
                self.logger.info("✅ Caching system initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Caching initialization failed: {e}")
                self.cache = None
        else:
            self.cache = None

    async def analyze_cross_timeframes(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: Optional[List[str]] = None
    ) -> OptimizedCrossTimeframeResult:
        """
        Perform optimized cross timeframe analysis.

        Args:
            data_dir: Data directory path
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to analyze (optional)

        Returns:
            OptimizedCrossTimeframeResult with comprehensive analysis results
        """
        start_time = time.time()

        if timeframes is None:
            timeframes = self.config.timeframes

        self.logger.info(f"⏰ Starting optimized cross timeframe analysis for {symbol} on {exchange} ({timeframes})")

        try:
            # Check cache first
            cache_key = f"cross_timeframe_{symbol}_{exchange}_{'_'.join(timeframes)}"
            if self.cache:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("✅ Using cached cross timeframe analysis result")
                    return cached_result

            # Memory checkpoint
            if self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("cross_timeframe_analysis_start"):
                    result = await self._perform_analysis(data_dir, symbol, exchange, timeframes)
            else:
                result = await self._perform_analysis(data_dir, symbol, exchange, timeframes)

            # Cache result
            if self.cache:
                await self.cache.set(cache_key, result)

            # Performance metrics
            execution_time = time.time() - start_time
            result.performance_metrics = {
                'execution_time_seconds': execution_time,
                'features_generated': len(result.cross_timeframe_features.columns),
                'selected_features_count': len(result.selected_features.get('final', [])),
                'memory_optimization_used': self.memory_optimizer is not None,
                'gpu_acceleration_used': self.gpu_manager is not None,
                'advanced_feature_selection_used': self.feature_selector is not None
            }

            self.logger.info(f"✅ Optimized cross timeframe analysis completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Optimized cross timeframe analysis failed: {e}")
            raise

    async def _perform_analysis(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: List[str]
    ) -> OptimizedCrossTimeframeResult:
        """Perform the actual analysis with optimizations."""

        # Load and validate data
        timeframe_data = await self._load_and_validate_data(data_dir, symbol, exchange, timeframes)

        # Align timeframes with optimization
        aligned_data = await self._align_timeframes_optimized(timeframe_data)

        # Engineer features with hardware acceleration
        cross_timeframe_features = await self._engineer_features_optimized(aligned_data)

        # Advanced feature selection
        selected_features = await self._perform_advanced_feature_selection(cross_timeframe_features)

        # Calculate metrics with parallel processing
        interaction_metrics = await self._calculate_interaction_metrics_optimized(aligned_data)
        timeframe_correlations = await self._calculate_timeframe_correlations_optimized(aligned_data)
        feature_importance = await self._calculate_feature_importance_optimized(cross_timeframe_features)

        # Financial and risk metrics
        financial_metrics, risk_metrics = await self._calculate_financial_risk_metrics(cross_timeframe_features)

        # Quality report
        quality_report = await self._generate_quality_report(timeframe_data, cross_timeframe_features)

        # Analysis metadata
        analysis_metadata = {
            'timeframes_analyzed': timeframes,
            'base_timeframe': self.config.base_timeframe,
            'total_features': len(cross_timeframe_features.columns),
            'selected_features': len(selected_features.get('final', [])),
            'interaction_features': self.config.interaction_features,
            'correlation_threshold': self.config.correlation_threshold,
            'optimizations_used': {
                'm1_optimizations': self.memory_optimizer is not None,
                'gpu_acceleration': self.gpu_manager is not None,
                'advanced_feature_selection': self.feature_selector is not None,
                'caching': self.cache is not None
            }
        }

        return OptimizedCrossTimeframeResult(
            cross_timeframe_features=cross_timeframe_features,
            selected_features=selected_features,
            interaction_metrics=interaction_metrics,
            timeframe_correlations=timeframe_correlations,
            feature_importance=feature_importance,
            financial_metrics=financial_metrics,
            risk_metrics=risk_metrics,
            quality_report=quality_report,
            analysis_metadata=analysis_metadata
        )
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
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
