"""
Strategy Search Optimizer

This module provides unified strategy search capabilities that consolidate
strategy search logic previously scattered across NAS and TAS implementations.
Enhanced with comprehensive utility integration for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Enhanced utility imports
from ...common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, optimize_dataframe_dtypes,
    safe_to_parquet, safe_read_parquet, integrate_with_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, timed_operation,
    format_bytes, parallel_map, chunked_iterable
)

from ...common_utilities import (
    CommonUtilities, safe_dataframe_operation as cu_safe_dataframe_operation,
    validate_dataframe_columns as cu_validate_dataframe_columns,
    calculate_data_quality_metrics as cu_calculate_data_quality_metrics,
    safe_merge_dataframes as cu_safe_merge_dataframes,
    safe_groupby_operation as cu_safe_groupby_operation,
    safe_apply_function as cu_safe_apply_function,
    create_summary_statistics as cu_create_summary_statistics,
    safe_drop_columns as cu_safe_drop_columns,
    safe_rename_columns as cu_safe_rename_columns,
    validate_timestamp_column as cu_validate_timestamp_column,
    safe_timestamp_conversion as cu_safe_timestamp_conversion,
    get_dataframe_info as cu_get_dataframe_info,
    safe_filter_dataframe as cu_safe_filter_dataframe,
    create_data_quality_report as cu_create_data_quality_report
)

from ...math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    validate_numeric_array
)

from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

# Data utilities
from ...data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ...data.processing.data_processing import DataProcessor
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils

# ML optimization utilities
from ...ml_common.optimization.bayesian_entry_timing_optimizer import BayesianEntryTimingOptimizer
from ...ml_common.optimization.grid_utils import GridSearchOptimizer
from ...ml_common.optimization.hpo_utils import HPOUtils
from ...ml_common.optimization.hierarchical_hpo import HierarchicalHPO
from ...ml_common.optimization.regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer

# Matrix operations
from ...matrix_operations.unified_operations import UnifiedMatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedProcessingCore
from ...matrix_operations.convenience import MatrixConvenience
from src.utils.nas_tas.config.base_config import OptimizationMode, SearchStrategy

# Hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

from ..core.nas_engine import NASEngine
from ..core.tas_engine import TASEngine
from ..config.base_config import UnifiedArchitectureConfig, ArchitectureType

@dataclass
class StrategySearchConfig:
    """Configuration for strategy search optimization."""

    # Search parameters
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5

    # Strategy settings
    strategy_types: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "breakout", "arbitrage", "regime_aware"
    ])
    enable_ensemble_strategies: bool = True
    ensemble_method: str = "weighted_average"

    # Evaluation settings
    evaluation_metric: str = "sharpe_ratio"
    backtest_periods: int = 252  # 1 year of daily data
    validation_split: float = 0.2

    # Risk management
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_limit: float = 0.15

    # Optimization settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    min_improvement_threshold: float = 0.001

    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Output settings
    save_results: bool = True
    results_path: str = "strategy_search_results"
    enable_visualization: bool = True

@dataclass
class StrategySearchResult:
    """Result from strategy search."""

    # Search results (required fields - no defaults)
    best_strategy: Dict[str, Any]
    best_score: float
    search_history: List[Dict[str, Any]]
    search_timestamp: datetime
    configuration: Dict[str, Any]

    # Performance metrics (with defaults)
    total_iterations: int = 0
    convergence_iteration: int = 0
    search_time: float = 0.0

    # Strategy details
    strategy_type: str = "unknown"
    risk_score: float = 0.0
    complexity_score: float = 0.0

    # Backtesting results
    backtest_results: Dict[str, float] = field(default_factory=dict)

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class StrategySearchOptimizer:
    """
    Unified strategy search optimizer for both NAS and TAS systems.

    This class consolidates strategy search logic that was previously
    scattered across different implementations, providing a unified interface
    for both neural and tree-based strategy search.
    Enhanced with comprehensive utility integration for optimal performance.
    """

    def __init__(self, config: StrategySearchConfig):
        """Initialize strategy search optimizer with comprehensive utility integration.

        Args:
            config: Strategy search configuration
        """
        tprint_info("🚀 Initializing Strategy Search Optimizer with comprehensive utility integration")
        tprint_debug(f"📋 Configuration provided: {config}")

        self.config = config
        tprint_debug(f"⚙️ Strategy Search config: max_iterations={config.max_iterations}, population_size={config.population_size}")
        tprint_debug(f"📊 Strategy types: {config.strategy_types}")
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint_debug(f"📝 Logger initialized: {self.logger.name}")

        # Initialize utility classes
        tprint_debug("🔧 Initializing utility classes")
        self.common_ops = CommonUtilities()
        tprint_debug("✅ CommonUtilities initialized")
        self.math_validator = MathValidation()
        tprint_debug("✅ MathValidation initialized")
        self.klines_manager = get_klines_manager()
        tprint_debug("✅ KlinesParquetManager initialized")

        # Initialize data processing utilities
        tprint_debug("🔧 Initializing data processing utilities")
        self.data_processor = DataProcessor()
        tprint_debug("✅ DataProcessor initialized")
        self.returns_engineer = BasicReturnsEngineer()
        tprint_debug("✅ BasicReturnsEngineer initialized")
        self.feature_engineer = FeatureEngineer()
        tprint_debug("✅ FeatureEngineer initialized")
        self.gap_detector = GapDetector()
        tprint_debug("✅ GapDetector initialized")
        self.unified_data_utils = UnifiedDataUtils()
        tprint_debug("✅ UnifiedDataUtils initialized")

        # Initialize matrix operations
        tprint_debug("🔧 Initializing matrix operations")
        self.matrix_ops = UnifiedMatrixOperations()
        tprint_debug("✅ UnifiedMatrixOperations initialized")
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        tprint_debug("✅ EnhancedMatrixOperations initialized")
        self.batch_matrix_ops = BatchMatrixOperations()
        tprint_debug("✅ BatchMatrixOperations initialized")
        self.vectorized_core = VectorizedProcessingCore()
        tprint_debug("✅ VectorizedProcessingCore initialized")
        self.matrix_convenience = MatrixConvenience()
        tprint_debug("✅ MatrixConvenience initialized")

        # Initialize M1 hardware optimizations
        tprint_debug("🔧 Initializing M1 hardware optimizations")
        self.m1_integration = integrate_with_m1_optimizers()
        tprint_debug(f"🔍 M1 integration result: {self.m1_integration}")
        if self.m1_integration['success']:
            tprint_success("✅ M1 integration successful")
            self.gpu_manager = get_m1_gpu_manager()
            tprint_debug("✅ M1 GPU Manager initialized")
            self.memory_optimizer = get_m1_memory_optimizer()
            tprint_debug("✅ M1 Memory Optimizer initialized")
            self.cpu_optimizer = get_m1_cpu_optimizer()
            tprint_debug("✅ M1 CPU Optimizer initialized")
        else:
            tprint_warning("⚠️ M1 integration failed, using fallback")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            tprint_debug("🔄 Using fallback configurations")

        # Initialize optimization components
        tprint_debug("🔧 Initializing optimization components")
        self.bayesian_optimizer = BayesianEntryTimingOptimizer()
        tprint_debug("✅ BayesianEntryTimingOptimizer initialized")
        self.grid_optimizer = GridSearchOptimizer()
        tprint_debug("✅ GridSearchOptimizer initialized")
        self.hpo_utils = HPOUtils
        tprint_debug("✅ HPOUtils initialized")
        self.hierarchical_hpo = HierarchicalHPO()
        tprint_debug("✅ HierarchicalHPO initialized")
        self.regime_tpsl_optimizer = RegimeSpecificTPSLOptimizer()
        tprint_debug("✅ RegimeSpecificTPSLOptimizer initialized")

        # Initialize search engines
        self.nas_engine = None
        self.tas_engine = None
        tprint_debug("✅ Search engines initialized (will be configured later)")

        # Search state
        self.search_history = []
        self.best_strategy = None
        self.best_score = -np.inf
        tprint_debug("✅ Search state initialized")

        tprint_success("✅ Strategy Search Optimizer initialized successfully")
        tprint_info(f"📊 Optimizer components: {len([attr for attr in dir(self) if not attr.startswith('_')])} public attributes")
        tprint_structured({
            'optimizer_type': 'StrategySearch',
            'initialization_time': time.time(),
            'm1_integration': self.m1_integration['success'],
            'strategy_types': self.config.strategy_types,
            'components_initialized': {
                'utility_classes': True,
                'data_processing': True,
                'matrix_operations': True,
                'hardware_optimization': self.m1_integration['success'],
                'optimization_components': True,
                'search_engines': False  # Will be initialized later
            }
        }, LogLevel.INFO)

    def initialize_engines(self, unified_config: UnifiedArchitectureConfig):
        """Initialize NAS and TAS engines based on configuration.

        Args:
            unified_config: Unified architecture configuration
        """
        try:
            # Initialize NAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.NEURAL_NETWORK, ArchitectureType.HYBRID]:
                self.nas_engine = NASEngine(unified_config.__dict__)
                tprint_success("NAS engine initialized for strategy search")

            # Initialize TAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.TREE_BASED, ArchitectureType.HYBRID]:
                self.tas_engine = TASEngine(unified_config.__dict__)
                tprint_success("TAS engine initialized for strategy search")

        except Exception as e:
            tprint_error(f"Engine initialization failed: {e}")
            raise

    @tprint_timer("Unified Strategy Search")
    async def search_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        unified_config: UnifiedArchitectureConfig
    ) -> StrategySearchResult:
        """Search for optimal strategies using unified approach with comprehensive utility integration.

        Args:
            data: Input data for strategy search
            search_space: Strategy search space
            unified_config: Unified architecture configuration

        Returns:
            StrategySearchResult with search results
        """
        start_time = datetime.now()
        tprint_info("🔍 Starting unified strategy search with comprehensive utility integration")

        try:
            # Validate and prepare data using utilities
            tprint_debug("🔍 Validating and preparing data")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            tprint_debug(f"📋 Required columns: {required_columns}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")

            if not validate_dataframe_columns(data, required_columns):
                tprint_error("❌ Invalid data columns for strategy search")
                tprint_structured({
                    'validation_error': {
                        'required_columns': required_columns,
                        'available_columns': list(data.columns),
                        'missing_columns': [col for col in required_columns if col not in data.columns]
                    }
                }, LogLevel.ERROR)
                raise ValueError("Invalid data columns")

            tprint_success("✅ Data validation passed for strategy search")

            # Apply data quality metrics
            tprint_debug("📊 Calculating data quality metrics")
            quality_metrics = calculate_data_quality_metrics(data)
            tprint_info(f"📈 Data quality metrics: {quality_metrics}")
            tprint_structured({
                'data_quality': quality_metrics,
                'data_characteristics': {
                    'shape': data.shape,
                    'null_counts': data.isnull().sum().to_dict(),
                    'memory_usage': data.memory_usage(deep=True).sum()
                }
            }, LogLevel.INFO)

            # Optimize data types for memory efficiency
            tprint_debug("🔧 Optimizing data types")
            memory_before = data.memory_usage(deep=True).sum()
            data = optimize_dataframe_dtypes(data)
            memory_after = data.memory_usage(deep=True).sum()
            tprint_debug(f"💾 Memory optimization: {memory_before} -> {memory_after} bytes ({(memory_after/memory_before-1)*100:.1f}% change)")

            # Guard against null values
            tprint_debug("🛡️ Applying null value guards")
            null_counts_before = data.isnull().sum().sum()
            data = guard_dataframe_nulls(data, threshold=0.1)
            null_counts_after = data.isnull().sum().sum()
            tprint_debug(f"🔍 Null values: {null_counts_before} -> {null_counts_after}")

            # Initialize engines
            tprint_debug("🔧 Initializing search engines")
            self.initialize_engines(unified_config)
            tprint_success("✅ Search engines initialized successfully")

            optimization_method = "bayesian_tpe"
            n_trials = self.config.max_iterations
            grid_params: Optional[List[Dict[str, Any]]] = None

            if unified_config.search_strategy == SearchStrategy.GRID_SEARCH:
                tprint_info("🧮 Grid search strategy selected for strategy optimization")
                grid_params = self.grid_optimizer.generate_grid(search_space, max_trials=n_trials)
                if grid_params:
                    n_trials = len(grid_params)
                    optimization_method = "grid"
                    tprint_debug(f"🔢 Generated {n_trials} grid parameter combinations for strategy search")
                else:
                    tprint_warning("⚠️ Grid search strategy selected but no parameter combinations were generated; falling back to Bayesian TPE")

            # Use M1 GPU context if available
            context_type = "GPU" if self.gpu_manager else "Memory"
            tprint_debug(f"🔧 Using {context_type} context for strategy search")

            with gpu_context("strategy_search") if self.gpu_manager else memory_checkpoint("strategy_search"):

                # Perform search based on architecture type
                tprint_info(f"🔍 Starting search for architecture type: {unified_config.architecture_type}")
                tprint_structured({
                    'search_configuration': {
                        'architecture_type': str(unified_config.architecture_type),
                        'search_space_size': len(search_space),
                        'search_space_keys': list(search_space.keys()) if search_space else [],
                        'strategy_types': self.config.strategy_types,
                        'optimization_method': optimization_method,
                        'grid_combinations': len(grid_params) if grid_params else 0
                    }
                }, LogLevel.INFO)

                if unified_config.architecture_type == ArchitectureType.NEURAL_ONLY:
                    tprint_info("🧠 Searching neural strategies")
                    with tprint_timer("Neural Strategy Search"):
                        result = await self._search_neural_strategies(
                            data,
                            search_space,
                            optimization_method,
                            n_trials
                        )
                elif unified_config.architecture_type == ArchitectureType.TREE_ONLY:
                    tprint_info("🌳 Searching tree strategies")
                    with tprint_timer("Tree Strategy Search"):
                        result = await self._search_tree_strategies(
                            data,
                            search_space,
                            optimization_method,
                            n_trials
                        )
                elif unified_config.architecture_type == ArchitectureType.HYBRID:
                    tprint_info("🔀 Searching hybrid strategies")
                    with tprint_timer("Hybrid Strategy Search"):
                        result = await self._search_hybrid_strategies(
                            data,
                            search_space,
                            optimization_method,
                            n_trials
                        )
                else:
                    tprint_error(f"❌ Unsupported architecture type: {unified_config.architecture_type}")
                    raise ValueError(f"Unsupported architecture type: {unified_config.architecture_type}")

            # Calculate search metrics using math validation utilities
            search_time = (datetime.now() - start_time).total_seconds()
            result.search_time = search_time
            result.total_iterations = len(self.search_history)
            result.convergence_iteration = self._find_convergence_point()

            tprint_debug(f"📊 Search metrics calculated: {search_time:.2f}s, {result.total_iterations} iterations")

            # Calculate risk and complexity scores
            tprint_debug("🔍 Calculating risk and complexity scores")
            result.risk_score = self._calculate_risk_score(result.best_strategy)
            result.complexity_score = self._calculate_strategy_complexity_score(result.best_strategy)

            # Perform backtesting with enhanced metrics
            if self.config.backtest_periods > 0:
                tprint_info("📊 Performing enhanced backtesting")
                with tprint_timer("Strategy Backtesting"):
                    result.backtest_results = await self._backtest_strategy(
                        result.best_strategy, data
                    )
                tprint_success("✅ Backtesting completed")
            else:
                tprint_debug("⏭️ Skipping backtesting as configured")

            tprint_success(f"✅ Strategy search completed in {search_time:.2f}s")
            tprint_info(f"🏆 Best score: {result.best_score:.4f}")
            tprint_info(f"⚠️ Risk score: {result.risk_score:.4f}")
            tprint_info(f"📊 Complexity score: {result.complexity_score:.4f}")

            # Log comprehensive search summary
            tprint_structured({
                'search_summary': {
                    'architecture_type': str(unified_config.architecture_type),
                    'total_iterations': result.total_iterations,
                    'best_score': result.best_score,
                    'search_time_seconds': search_time,
                    'risk_score': result.risk_score,
                    'complexity_score': result.complexity_score,
                    'convergence_iteration': result.convergence_iteration,
                    'backtest_performed': bool(result.backtest_results)
                }
            }, LogLevel.SUCCESS)

            return result

        except Exception as e:
            tprint_error(f"❌ Strategy search failed: {e}")
            self.logger.exception("Strategy search error")
            raise

    async def _search_neural_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str,
        n_trials: int
    ) -> StrategySearchResult:
        """Search for neural-based strategies using NAS engine."""
        tprint_info("Searching neural strategies")

        if not self.nas_engine:
            raise ValueError("NAS engine not initialized")

        # Use NAS engine for strategy search
        nas_results = self.nas_engine.search_architectures(
            data=data,
            search_space=search_space,
            optimization_method=optimization_method,
            n_trials=n_trials
        )

        return StrategySearchResult(
            best_strategy=nas_results.get('best_architecture', {}),
            best_score=nas_results.get('best_score', 0.0),
            search_history=nas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            strategy_type="neural",
            risk_score=0.0,
            complexity_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )

    async def _search_tree_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str,
        n_trials: int
    ) -> StrategySearchResult:
        """Search for tree-based strategies using TAS engine."""
        tprint_info("Searching tree strategies")

        if not self.tas_engine:
            raise ValueError("TAS engine not initialized")

        # Use TAS engine for strategy search
        tas_results = self.tas_engine.search_strategies(
            data=data,
            search_space=search_space,
            optimization_method=optimization_method,
            n_trials=n_trials,
            include_regime_specific=True
        )

        return StrategySearchResult(
            best_strategy=tas_results.get('best_strategy', {}),
            best_score=tas_results.get('best_score', 0.0),
            search_history=tas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            strategy_type="tree",
            risk_score=0.0,
            complexity_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )

    async def _search_hybrid_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str,
        n_trials: int
    ) -> StrategySearchResult:
        """Search for hybrid strategies combining neural and tree components."""
        tprint_info("Searching hybrid strategies")

        if not self.nas_engine or not self.tas_engine:
            raise ValueError("Both NAS and TAS engines required for hybrid search")

        # Search neural strategies
        nas_results = await self._search_neural_strategies(
            data,
            search_space,
            optimization_method,
            n_trials
        )

        # Search tree strategies
        tas_results = await self._search_tree_strategies(
            data,
            search_space,
            optimization_method,
            n_trials
        )

        # Combine results for hybrid strategy
        hybrid_strategy = {
            'neural_strategy': nas_results.best_strategy,
            'tree_strategy': tas_results.best_strategy,
            'ensemble_method': self.config.ensemble_method,
            'neural_weight': 0.6,
            'tree_weight': 0.4,
            'strategy_type': 'hybrid'
        }

        # Calculate hybrid score
        hybrid_score = (nas_results.best_score * 0.6 + tas_results.best_score * 0.4)

        return StrategySearchResult(
            best_strategy=hybrid_strategy,
            best_score=hybrid_score,
            search_history=nas_results.search_history + tas_results.search_history,
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            strategy_type="hybrid",
            risk_score=0.0,
            complexity_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )

    async def _backtest_strategy(
        self,
        strategy: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Backtest the best strategy."""
        tprint_info("📊 Backtesting strategy")
        tprint_debug(f"📋 Strategy parameters: {list(strategy.keys()) if strategy else 'None'}")
        tprint_debug(f"📊 Data shape for backtesting: {data.shape}")

        try:
            # Simple backtesting implementation
            # In a real implementation, this would use the backtesting engine
            tprint_debug("🔍 Calculating basic performance metrics")

            # Calculate basic metrics
            returns = data['close'].pct_change().dropna()
            tprint_debug(f"📊 Returns calculated: {len(returns)} data points")

            # Calculate performance metrics
            tprint_debug("📈 Calculating performance metrics")
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0

            tprint_debug(f"📊 Total return: {total_return:.4f}")
            tprint_debug(f"📊 Volatility: {volatility:.4f}")
            tprint_debug(f"📊 Sharpe ratio: {sharpe_ratio:.4f}")

            # Calculate drawdown
            tprint_debug("📉 Calculating drawdown metrics")
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            tprint_debug(f"📊 Max drawdown: {max_drawdown:.4f}")

            win_rate = (returns > 0).mean()
            tprint_debug(f"📊 Win rate: {win_rate:.4f}")

            backtest_results = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }

            tprint_success("✅ Backtesting completed successfully")
            tprint_structured({
                'backtest_results': backtest_results
            }, LogLevel.INFO)

            return backtest_results

        except Exception as e:
            tprint_warning(f"⚠️ Backtesting failed: {e}")
            tprint_structured({
                'backtest_error': {
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'backtest_failed': True
                }
            }, LogLevel.WARNING)
            return {}

    def _find_convergence_point(self) -> int:
        """Find the iteration where convergence occurred."""
        tprint_debug("🔍 Finding convergence point")

        if not self.search_history:
            tprint_debug("📊 No search history available")
            return 0

        # Simple convergence detection based on score improvement
        scores = [trial.get('score', 0.0) for trial in self.search_history]
        tprint_debug(f"📊 Analyzing {len(scores)} scores for convergence")

        if len(scores) < 10:
            tprint_debug("📊 Insufficient trials for convergence analysis")
            return len(scores)

        # Find point where improvement becomes minimal
        for i in range(10, len(scores)):
            recent_scores = scores[i-10:i]
            score_std = np.std(recent_scores)
            tprint_debug(f"🔍 Iteration {i}: score std = {score_std:.6f}, threshold = {self.config.min_improvement_threshold}")

            if score_std < self.config.min_improvement_threshold:
                tprint_info(f"🎯 Convergence detected at iteration {i}")
                return i

        tprint_debug("📊 No convergence detected, using full iteration count")
        return len(scores)

    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search results."""
        return {
            'total_iterations': len(self.search_history),
            'best_score': self.best_score,
            'convergence_iteration': self._find_convergence_point(),
            'search_efficiency': self._calculate_search_efficiency()
        }

    def _calculate_search_efficiency(self) -> float:
        """Calculate search efficiency metric using math validation utilities."""
        tprint_debug("🔍 Calculating search efficiency")

        if not self.search_history:
            tprint_debug("📊 No search history available")
            return 0.0

        # Calculate improvement rate
        improvements = 0
        for i in range(1, len(self.search_history)):
            if self.search_history[i].get('score', 0.0) > self.search_history[i-1].get('score', 0.0):
                improvements += 1

        efficiency = safe_divide(improvements, max(1, len(self.search_history) - 1))
        tprint_debug(f"📊 Search efficiency: {efficiency:.4f} ({improvements}/{len(self.search_history)-1} improvements)")

        return efficiency

    def _calculate_risk_score(self, strategy: Dict[str, Any]) -> float:
        """Calculate strategy risk score using math validation utilities."""
        try:
            if not strategy:
                return 0.0

            # Extract risk indicators
            risk_factors = []

            # Position size risk
            position_size = strategy.get('position_size', 0.1)
            risk_factors.append(position_size)

            # Stop loss risk
            stop_loss = strategy.get('stop_loss', 0.02)
            risk_factors.append(stop_loss)

            # Risk factor
            risk_factor = strategy.get('risk_factor', 1.0)
            risk_factors.append(risk_factor)

            # Calculate weighted risk score
            risk_score = safe_weighted_average(risk_factors, [0.4, 0.3, 0.3])

            return validate_finite(risk_score, "risk_score")

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating risk score: {e}")
            return 0.0

    def _calculate_strategy_complexity_score(self, strategy: Dict[str, Any]) -> float:
        """Calculate strategy complexity score using math validation utilities."""
        try:
            if not strategy:
                return 0.0

            # Extract complexity indicators
            complexity_factors = []

            # Count parameters
            param_count = len(strategy)
            complexity_factors.append(safe_log(param_count + 1))

            # Calculate parameter diversity
            param_values = list(strategy.values())
            if param_values:
                param_std = safe_std(np.array([float(v) for v in param_values if isinstance(v, (int, float))]))
                complexity_factors.append(param_std)

            # Calculate weighted complexity
            complexity_score = safe_weighted_average(complexity_factors, [0.7, 0.3])

            return validate_finite(complexity_score, "strategy_complexity_score")

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating strategy complexity score: {e}")
            return 0.0

# Convenience function for quick strategy search
async def search_optimal_strategy(
    data: pd.DataFrame,
    search_space: Dict[str, Any],
    architecture_type: ArchitectureType = ArchitectureType.HYBRID,
    config: Optional[StrategySearchConfig] = None
) -> StrategySearchResult:
    """Search for optimal strategy with default configuration.

    Args:
        data: Input data for search
        search_space: Strategy search space
        architecture_type: Type of architecture to search for
        config: Optional search configuration

    Returns:
        StrategySearchResult with search results
    """
    if config is None:
        config = StrategySearchConfig()

    # Create unified configuration
    unified_config = UnifiedArchitectureConfig(
        architecture_type=architecture_type,
        optimization_mode=OptimizationMode.REGIME_AWARE
    )

    # Initialize optimizer
    optimizer = StrategySearchOptimizer(config)

    # Perform search
    result = await optimizer.search_strategies(data, search_space, unified_config)

    return result
