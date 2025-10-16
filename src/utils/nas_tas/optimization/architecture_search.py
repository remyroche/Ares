"""
Architecture Search Optimizer

This module provides unified architecture search capabilities that consolidate
search logic previously scattered across NAS and TAS implementations.
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

# Hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

from ..core.nas_engine import NASEngine
from ..core.tas_engine import TASEngine
from ..config.base_config import UnifiedArchitectureConfig, ArchitectureType, SearchStrategy

@dataclass
class ArchitectureSearchConfig:
    """Configuration for architecture search optimization."""

    # Search parameters
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5

    # Evaluation settings
    evaluation_metric: str = "f1_score"
    validation_split: float = 0.2
    cv_folds: int = 5

    # Optimization settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    min_improvement_threshold: float = 0.001

    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Output settings
    save_results: bool = True
    results_path: str = "architecture_search_results"
    enable_visualization: bool = True

@dataclass
class ArchitectureSearchResult:
    """Result from architecture search."""

    # Search results
    best_architecture: Dict[str, Any]
    best_score: float
    search_history: List[Dict[str, Any]]

    # Performance metrics
    total_iterations: int
    convergence_iteration: int
    search_time: float

    # Architecture details
    architecture_type: ArchitectureType
    complexity_score: float
    efficiency_score: float

    # Metadata
    search_timestamp: datetime
    configuration: Dict[str, Any]

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class ArchitectureSearchOptimizer:
    """
    Unified architecture search optimizer for both NAS and TAS systems.

    This class consolidates architecture search logic that was previously
    scattered across different implementations, providing a unified interface
    for both neural and tree architecture search.
    Enhanced with comprehensive utility integration for optimal performance.
    """

    def __init__(self, config: ArchitectureSearchConfig):
        """Initialize architecture search optimizer with comprehensive utility integration.

        Args:
            config: Architecture search configuration
        """
        tprint_info("🚀 Initializing Architecture Search Optimizer with comprehensive utility integration")
        tprint_debug(f"📋 Configuration provided: {config}")

        self.config = config
        tprint_debug(f"⚙️ Architecture Search config: max_iterations={config.max_iterations}, population_size={config.population_size}")
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
        tprint_debug("✅ MatrixOperations initialized")
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        tprint_debug("✅ EnhancedMatrixOperations initialized")
        self.batch_matrix_ops = BatchMatrixOperations()
        tprint_debug("✅ BatchMatrixOperations initialized")
        self.vectorized_core = VectorizedProcessingCore()
        tprint_debug("✅ VectorizedCore initialized")
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
        self.best_architecture = None
        self.best_score = -np.inf
        tprint_debug("✅ Search state initialized")

        tprint_success("✅ Architecture Search Optimizer initialized successfully")
        tprint_info(f"📊 Optimizer components: {len([attr for attr in dir(self) if not attr.startswith('_')])} public attributes")
        tprint_structured({
            'optimizer_type': 'ArchitectureSearch',
            'initialization_time': time.time(),
            'm1_integration': self.m1_integration['success'],
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
            if unified_config.architecture_type in [ArchitectureType.NEURAL_ONLY, ArchitectureType.HYBRID_NEURAL_TREE]:
                self.nas_engine = NASEngine(unified_config.__dict__)
                tprint_success("NAS engine initialized")

            # Initialize TAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.TREE_ONLY, ArchitectureType.HYBRID_NEURAL_TREE]:
                self.tas_engine = TASEngine(unified_config.__dict__)
                tprint_success("TAS engine initialized")

        except Exception as e:
            tprint_error(f"Engine initialization failed: {e}")
            raise

    @tprint_timer("Unified Architecture Search")
    async def search_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        unified_config: UnifiedArchitectureConfig
    ) -> ArchitectureSearchResult:
        """Search for optimal architectures using unified approach with comprehensive utility integration.

        Args:
            data: Input data for architecture search
            search_space: Architecture search space
            unified_config: Unified architecture configuration

        Returns:
            ArchitectureSearchResult with search results
        """
        start_time = datetime.now()
        tprint_info("🔍 Starting unified architecture search with comprehensive utility integration")

        try:
            # Validate and prepare data using utilities
            tprint_debug("🔍 Validating and preparing data")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            tprint_debug(f"📋 Required columns: {required_columns}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")

            if not validate_dataframe_columns(data, required_columns):
                tprint_error("❌ Invalid data columns for architecture search")
                tprint_structured({
                    'validation_error': {
                        'required_columns': required_columns,
                        'available_columns': list(data.columns),
                        'missing_columns': [col for col in required_columns if col not in data.columns]
                    }
                }, LogLevel.ERROR)
                raise ValueError("Invalid data columns")

            tprint_success("✅ Data validation passed for architecture search")

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

            # Use M1 GPU context if available
            context_type = "GPU" if self.gpu_manager else "Memory"
            tprint_debug(f"🔧 Using {context_type} context for architecture search")

            optimization_method = "bayesian_tpe"
            n_trials = self.config.max_iterations
            grid_params: Optional[List[Dict[str, Any]]] = None

            if unified_config.search_strategy == SearchStrategy.GRID_SEARCH:
                tprint_info("🧮 Grid search strategy selected for architecture optimization")
                grid_params = self.grid_optimizer.generate_grid(search_space, max_trials=n_trials)
                if grid_params:
                    n_trials = len(grid_params)
                    optimization_method = "grid"
                    tprint_debug(f"🔢 Generated {n_trials} grid parameter combinations")
                else:
                    tprint_warning("⚠️ Grid search strategy selected but no parameter combinations were generated; falling back to Bayesian TPE")

            with gpu_context("architecture_search") if self.gpu_manager else memory_checkpoint("architecture_search"):

                # Perform search based on architecture type
                tprint_info(f"🔍 Starting search for architecture type: {unified_config.architecture_type}")
                tprint_structured({
                    'search_configuration': {
                        'architecture_type': str(unified_config.architecture_type),
                        'search_space_size': len(search_space),
                        'search_space_keys': list(search_space.keys()) if search_space else [],
                        'optimization_method': optimization_method,
                        'grid_combinations': len(grid_params) if grid_params else 0
                    }
                }, LogLevel.INFO)

                if unified_config.architecture_type == ArchitectureType.NEURAL_ONLY:
                    tprint_info("🧠 Searching neural architectures")
                    with tprint_timer("Neural Architecture Search"):
                        result = await self._search_neural_architectures(
                            data,
                            search_space,
                            optimization_method,
                            n_trials
                        )
                elif unified_config.architecture_type == ArchitectureType.TREE_ONLY:
                    tprint_info("🌳 Searching tree architectures")
                    with tprint_timer("Tree Architecture Search"):
                        result = await self._search_tree_architectures(
                            data,
                            search_space,
                            optimization_method,
                            n_trials
                        )
                elif unified_config.architecture_type == ArchitectureType.HYBRID_NEURAL_TREE:
                    tprint_info("🔀 Searching hybrid architectures")
                    with tprint_timer("Hybrid Architecture Search"):
                        result = await self._search_hybrid_architectures(
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

            # Calculate complexity and efficiency scores
            tprint_debug("🔍 Calculating complexity and efficiency scores")
            result.complexity_score = self._calculate_complexity_score(result.best_architecture)
            result.efficiency_score = self._calculate_efficiency_score(result.best_architecture, result.best_score)

            tprint_success(f"✅ Architecture search completed in {search_time:.2f}s")
            tprint_info(f"🏆 Best score: {result.best_score:.4f}")
            tprint_info(f"📊 Complexity score: {result.complexity_score:.4f}")
            tprint_info(f"⚡ Efficiency score: {result.efficiency_score:.4f}")

            # Log comprehensive search summary
            tprint_structured({
                'search_summary': {
                    'architecture_type': str(unified_config.architecture_type),
                    'total_iterations': result.total_iterations,
                    'best_score': result.best_score,
                    'search_time_seconds': search_time,
                    'complexity_score': result.complexity_score,
                    'efficiency_score': result.efficiency_score,
                    'convergence_iteration': result.convergence_iteration
                }
            }, LogLevel.SUCCESS)

            return result

        except Exception as e:
            tprint_error(f"❌ Architecture search failed: {e}")
            self.logger.exception("Architecture search error")
            raise

    async def _search_neural_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str,
        n_trials: int
    ) -> ArchitectureSearchResult:
        """Search for neural architectures using NAS engine."""
        tprint_info("🔍 Searching neural architectures")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"🔧 Search space size: {len(search_space)}")

        if not self.nas_engine:
            tprint_error("❌ NAS engine not initialized")
            raise ValueError("NAS engine not initialized")

        tprint_debug("✅ NAS engine is available")

        # Use NAS engine for architecture search
        tprint_debug(f"🔧 Starting NAS engine search with method={optimization_method}, trials={n_trials}")
        nas_results = self.nas_engine.search_architectures(
            data=data,
            search_space=search_space,
            optimization_method=optimization_method,
            n_trials=n_trials
        )

        tprint_debug(f"📊 NAS search completed: {len(nas_results.get('trials', []))} trials")
        tprint_debug(f"🏆 Best score: {nas_results.get('best_score', 0.0)}")

        result = ArchitectureSearchResult(
            best_architecture=nas_results.get('best_architecture', {}),
            best_score=nas_results.get('best_score', 0.0),
            search_history=nas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            architecture_type=ArchitectureType.NEURAL_ONLY,
            complexity_score=0.0,
            efficiency_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )

        tprint_success("✅ Neural architecture search completed")
        tprint_structured({
            'neural_search_result': {
                'best_architecture_found': bool(result.best_architecture),
                'best_score': result.best_score,
                'trials_completed': len(result.search_history),
                'architecture_type': 'NEURAL_ONLY'
            }
        }, LogLevel.INFO)

        return result

    async def _search_tree_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str,
        n_trials: int
    ) -> ArchitectureSearchResult:
        """Search for tree architectures using TAS engine."""
        tprint_info("🔍 Searching tree architectures")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"🔧 Search space size: {len(search_space)}")

        if not self.tas_engine:
            tprint_error("❌ TAS engine not initialized")
            raise ValueError("TAS engine not initialized")

        tprint_debug("✅ TAS engine is available")

        # Use TAS engine for strategy search
        tprint_debug(f"🔧 Starting TAS engine search with method={optimization_method}, trials={n_trials}")
        tas_results = self.tas_engine.search_strategies(
            data=data,
            search_space=search_space,
            optimization_method=optimization_method,
            n_trials=n_trials,
            include_regime_specific=True
        )

        tprint_debug(f"📊 TAS search completed: {len(tas_results.get('trials', []))} trials")
        tprint_debug(f"🏆 Best score: {tas_results.get('best_score', 0.0)}")

        result = ArchitectureSearchResult(
            best_architecture=tas_results.get('best_strategy', {}),
            best_score=tas_results.get('best_score', 0.0),
            search_history=tas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            architecture_type=ArchitectureType.TREE_ONLY,
            complexity_score=0.0,
            efficiency_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )

        tprint_success("✅ Tree architecture search completed")
        tprint_structured({
            'tree_search_result': {
                'best_architecture_found': bool(result.best_architecture),
                'best_score': result.best_score,
                'trials_completed': len(result.search_history),
                'architecture_type': 'TREE_ONLY'
            }
        }, LogLevel.INFO)

        return result

    async def _search_hybrid_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str,
        n_trials: int
    ) -> ArchitectureSearchResult:
        """Search for hybrid architectures combining neural and tree components."""
        tprint_info("🔍 Searching hybrid architectures")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"🔧 Search space size: {len(search_space)}")

        if not self.nas_engine or not self.tas_engine:
            tprint_error("❌ Both NAS and TAS engines required for hybrid search")
            tprint_debug(f"🔍 NAS engine available: {bool(self.nas_engine)}")
            tprint_debug(f"🔍 TAS engine available: {bool(self.tas_engine)}")
            raise ValueError("Both NAS and TAS engines required for hybrid search")

        tprint_debug("✅ Both NAS and TAS engines are available")

        # Search neural architectures
        tprint_info("🧠 Starting neural architecture search for hybrid")
        nas_results = await self._search_neural_architectures(
            data,
            search_space,
            optimization_method,
            n_trials
        )

        # Search tree architectures
        tprint_info("🌳 Starting tree architecture search for hybrid")
        tas_results = await self._search_tree_architectures(
            data,
            search_space,
            optimization_method,
            n_trials
        )

        # Combine results for hybrid architecture
        tprint_debug("🔀 Combining neural and tree results for hybrid architecture")
        hybrid_architecture = {
            'neural_config': nas_results.best_architecture,
            'tree_config': tas_results.best_architecture,
            'ensemble_method': 'weighted_average',
            'neural_weight': 0.6,
            'tree_weight': 0.4
        }

        # Calculate hybrid score
        tprint_debug(f"📊 Calculating hybrid score from neural: {nas_results.best_score:.4f}, tree: {tas_results.best_score:.4f}")
        hybrid_score = (nas_results.best_score * 0.6 + tas_results.best_score * 0.4)
        tprint_debug(f"🏆 Hybrid score: {hybrid_score:.4f}")

        result = ArchitectureSearchResult(
            best_architecture=hybrid_architecture,
            best_score=hybrid_score,
            search_history=nas_results.search_history + tas_results.search_history,
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            architecture_type=ArchitectureType.HYBRID_NEURAL_TREE,
            complexity_score=0.0,
            efficiency_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )

        tprint_success("✅ Hybrid architecture search completed")
        tprint_structured({
            'hybrid_search_result': {
                'neural_score': nas_results.best_score,
                'tree_score': tas_results.best_score,
                'hybrid_score': hybrid_score,
                'neural_trials': len(nas_results.search_history),
                'tree_trials': len(tas_results.search_history),
                'total_trials': len(result.search_history),
                'architecture_type': 'HYBRID_NEURAL_TREE'
            }
        }, LogLevel.INFO)

        return result

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
        if not self.search_history:
            return 0.0

        # Calculate improvement rate
        improvements = 0
        for i in range(1, len(self.search_history)):
            if self.search_history[i].get('score', 0.0) > self.search_history[i-1].get('score', 0.0):
                improvements += 1

        return safe_divide(improvements, max(1, len(self.search_history) - 1))

    def _calculate_complexity_score(self, architecture: Dict[str, Any]) -> float:
        """Calculate architecture complexity score using math validation utilities."""
        try:
            if not architecture:
                return 0.0

            # Extract complexity indicators
            complexity_factors = []

            # Count parameters
            param_count = len(architecture)
            complexity_factors.append(safe_log(param_count + 1))

            # Calculate parameter diversity
            param_values = list(architecture.values())
            if param_values:
                param_std = safe_std(np.array([float(v) for v in param_values if isinstance(v, (int, float))]))
                complexity_factors.append(param_std)

            # Calculate weighted complexity
            complexity_score = safe_weighted_average(complexity_factors, [0.7, 0.3])

            return validate_finite(complexity_score, "complexity_score")

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating complexity score: {e}")
            return 0.0

    def _calculate_efficiency_score(self, architecture: Dict[str, Any], performance_score: float) -> float:
        """Calculate architecture efficiency score using math validation utilities."""
        try:
            if not architecture or performance_score <= 0:
                return 0.0

            # Calculate complexity score
            complexity = self._calculate_complexity_score(architecture)

            # Efficiency = Performance / Complexity (higher is better)
            if complexity > 0:
                efficiency = safe_divide(performance_score, complexity)
            else:
                efficiency = performance_score

            return validate_finite(efficiency, "efficiency_score")

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating efficiency score: {e}")
            return 0.0

# Convenience function for quick architecture search
async def search_optimal_architecture(
    data: pd.DataFrame,
    search_space: Dict[str, Any],
    architecture_type: ArchitectureType = ArchitectureType.HYBRID_NEURAL_TREE,
    config: Optional[ArchitectureSearchConfig] = None
) -> ArchitectureSearchResult:
    """Search for optimal architecture with default configuration.

    Args:
        data: Input data for search
        search_space: Architecture search space
        architecture_type: Type of architecture to search for
        config: Optional search configuration

    Returns:
        ArchitectureSearchResult with search results
    """
    if config is None:
        config = ArchitectureSearchConfig()

    # Create unified configuration
    unified_config = UnifiedArchitectureConfig(
        architecture_type=architecture_type,
        optimization_mode=OptimizationMode.REGIME_AWARE
    )

    # Initialize optimizer
    optimizer = ArchitectureSearchOptimizer(config)

    # Perform search
    result = await optimizer.search_architectures(data, search_space, unified_config)

    return result
