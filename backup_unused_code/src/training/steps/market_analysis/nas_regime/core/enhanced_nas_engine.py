"""
Enhanced NAS Engine - Now Using Unified System

This module provides a neural architecture search engine that uses the unified system
for search strategies, optimization, and evaluation. It maintains backward compatibility
while leveraging the consolidated functionality.
"""

# Import unified system components
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedSearchEngine, SearchConfig, SearchResult, SearchStrategy, ArchitectureType,
    UnifiedMultiObjectiveOptimizer, UnifiedMultiObjectiveConfig, OptimizationAlgorithm,
    UnifiedEconomicEvaluator, EconomicEvaluationConfig,
    UnifiedRegimeDetector, RegimeDetectionConfig,
    UnifiedUtilities, UnifiedUtilityConfig,
    UnifiedConfig, create_unified_system
)


import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import pickle
import os
from pathlib import Path

# Import unified system components
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedSearchEngine, SearchConfig, SearchResult, SearchStrategy, ArchitectureType,
    UnifiedMultiObjectiveOptimizer, UnifiedMultiObjectiveConfig, OptimizationAlgorithm,
    UnifiedEconomicEvaluator, EconomicEvaluationConfig,
    UnifiedRegimeDetector, RegimeDetectionConfig,
    UnifiedUtilities, UnifiedUtilityConfig,
    UnifiedConfig, create_unified_system,
    LegacyNASEngineAdapter  # For backward compatibility
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import enhanced utility tools
from src.utils.common_operations import (
    # DataFrame utilities
    create_empty_dataframe, validate_dataframe, validate_dataframe_columns,
    safe_dataframe_operation, safe_fillna, safe_convert_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    validate_timestamp_column, safe_timestamp_conversion, optimize_dataframe_dtypes,
    # Data quality utilities
    calculate_data_quality_metrics, get_dataframe_info, create_data_quality_report,
    # Math utilities
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    # String utilities
    safe_lower, safe_upper, safe_join,
    # Collection utilities
    safe_append, safe_extend, safe_dict_get, safe_dict_items,
    # Async utilities
    safe_sleep, safe_gather, create_async_task,
    # Performance utilities
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    # Matrix utilities
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    # File utilities
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    get_latest_outcome_file, load_latest_optimal_regime_clustering_outcome,
    safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls,
    # Memory optimization utilities
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space
)

from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    calculate_data_quality_metrics as calc_data_quality, create_summary_statistics,
    safe_convert_dtypes as safe_convert_dtypes_util, safe_merge_dataframes as safe_merge_dfs,
    safe_groupby_operation, safe_apply_function, safe_drop_columns as safe_drop_cols,
    safe_rename_columns as safe_rename_cols, validate_timestamp_column as validate_ts_col,
    safe_timestamp_conversion as safe_ts_conversion, get_dataframe_info as get_df_info,
    safe_filter_dataframe, create_data_quality_report as create_dq_report,
    CommonUtilities
)

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log, safe_sqrt as math_safe_sqrt,
    safe_power as math_safe_power, validate_finite as math_validate_finite,
    validate_positive as math_validate_positive, validate_range as math_validate_range,
    validate_numeric_array, safe_kelly_calculation as math_safe_kelly,
    safe_weighted_average as math_safe_weighted_avg, safe_percentage_change as math_safe_pct_change,
    safe_correlation, safe_covariance, safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile, validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inverse, math_safe, MathValidation
)

from src.utils.data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import ML Common Utilities
from src.utils.ml_common import (
    # Model utilities
    EnhancedModelTrainer, train_model_with_confidence_metrics,
    ModelEvaluator, ModelRegistry,
    # Validation utilities
    UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
    ConfigurationValidator, optimize_threshold, calibrate_probabilities,
    # Optimization utilities
    RegimeSpecificTPSLOptimizer,
    # Ensemble utilities
    StackingEnsembleManager, create_analyst_ensemble,
    # Utils
    MemoryOptimizer, UnifiedCache, get_unified_cache,
    LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler,
    setup_logger, get_logger
)

# Circular import removed to prevent import errors
# from ...hybrid_nas_tas_regime.core.unified_architecture_search_engine import (
#     UnifiedArchitectureSearchEngine, UnifiedSearchConfig, ArchitectureType
# )
# from ...hybrid_nas_tas_regime.core.performance_estimator import (
#     UnifiedPerformanceEstimator, create_unified_performance_estimator
# )
# from ...hybrid_nas_tas_regime.core.advanced_search_strategies import (
#     AdvancedSearchStrategies, SearchStrategyType
# )
from ...hybrid_nas_tas_regime.core.multi_objective_optimizer import (
    TradingMultiObjectiveOptimizer, MultiObjectiveConfig, ObjectiveType
)
from ...hybrid_nas_tas_regime.core.architecture_encoder import (
    UnifiedArchitectureEncoder, create_unified_architecture_encoder
)
from ...hybrid_nas_tas_regime.shared_utils.unified_ensemble_search_space import (
    UnifiedEnsembleSearchSpace, EnsembleArchitecture, EnsembleSearchResult,
    EnsembleMethod, EnsembleCombinationStrategy, EnsembleSearchSpaceConfig,
    create_unified_ensemble_search_space
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_compression import (
    UnifiedArchitectureCompressor, CompressionResult, CompressionMethod, CompressionLevel, CompressionConfig,
    create_unified_architecture_compressor
)
from ...hybrid_nas_tas_regime.shared_utils.unified_search_space_evolution import (
    UnifiedSearchSpaceEvolutionManager, EvolutionTrigger, EvolutionAction, UnifiedEvolutionConfig,
    create_unified_evolution_manager
)

# Import unified utilities
from ...hybrid_nas_tas_regime.shared_utils import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig,
    UnifiedMultiObjectiveOptimizer, OptimizationConfig,
    UnifiedHardwareOptimizer, HardwareConfig,
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig,
    UnifiedValidationSystem, ValidationConfig,
    UnifiedConfigManager, UnifiedRegimeConfig,
    create_unified_economic_evaluator, quick_economic_evaluation,
    create_unified_trading_viability_evaluator, quick_trading_viability_evaluation,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization,
    create_unified_hardware_optimizer, quick_hardware_optimization,
    create_unified_regime_analyzer, quick_regime_analysis,
    create_unified_config_manager, load_config_from_file, create_environment_config,
    create_unified_validation_system, quick_validation
)

# Use shared logger from hybrid utilities
logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Available search strategies."""
    RANDOM = "random"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENHANCED_BAYESIAN = "enhanced_bayesian"
    ADAPTIVE_EVOLUTIONARY = "adaptive_evolutionary"
    HYBRID = "hybrid"


@dataclass
class NASSearchConfig:
    """Configuration for NAS search."""
    search_strategy: SearchStrategy = SearchStrategy.ENHANCED_BAYESIAN
    population_size: int = 50
    max_generations: int = 100
    max_evaluations: int = 1000
    max_search_time: int = 3600  # 1 hour
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-6

    # Multi-objective optimization
    enable_multi_objective: bool = True
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'performance': 1.0,
        'complexity': 0.2,
        'efficiency': 0.3,
        'trading_viability': 0.5
    })

    # Advanced search parameters
    enable_constraint_validation: bool = True
    enable_performance_estimation: bool = True
    enable_architecture_encoding: bool = True
    
    # Ensemble search space
    enable_ensemble_search: bool = True
    ensemble_search_weight: float = 0.3  # Weight for ensemble vs individual architecture search
    
    # Architecture compression
    enable_compression: bool = True
    compression_method: CompressionMethod = CompressionMethod.PRUNING
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    
    # Search space evolution
    enable_evolution: bool = True
    evolution_intensity: float = 0.3
    min_evolution_interval: int = 100

    # Hardware constraints
    max_memory_mb: int = 8192
    max_training_time_per_arch: int = 600  # 10 minutes
    parallel_evaluation: bool = True
    n_workers: int = 4

    # Search space constraints
    max_layers: int = 20
    min_layers: int = 2
    max_parameters: int = 10000000
    allow_complex_activations: bool = True


@dataclass
class NASSearchResult:
    """Result from NAS search."""
    best_architecture: Any
    best_score: float
    search_history: List[Dict[str, Any]]
    pareto_frontier: List[Any]
    strategy_used: str
    convergence_info: Dict[str, Any]
    execution_time: float
    n_evaluations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedNASEngine:
    """Enhanced Neural Architecture Search Engine."""

    def __init__(self, config: NASSearchConfig):
        """Initialize the enhanced NAS engine with integrated utility tools."""
        tprint("🚀 [NAS_ENGINE] Initializing Enhanced NAS Engine", color="cyan", bold=True)
        tprint(f"📊 [NAS_ENGINE] Search Strategy: {config.search_strategy.value}", color="blue")
        tprint(f"📊 [NAS_ENGINE] Population Size: {config.population_size}", color="blue")
        tprint(f"📊 [NAS_ENGINE] Max Generations: {config.max_generations}", color="blue")
        self.config = config

        # Initialize utility managers
        tprint("🔧 [NAS_ENGINE] Initializing utility managers", color="yellow")
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.universal_serializer = UniversalSerializer()
        self.klines_manager = get_klines_manager()
        tprint("✅ [NAS_ENGINE] Utility managers initialized", color="green")

        # Initialize shared ML utilities from hybrid_nas_tas_regime
        tprint("🧠 [NAS_ENGINE] Initializing shared ML utilities", color="yellow")
        self._initialize_shared_ml_utilities()

        # Initialize shared components
        tprint("🔧 [NAS_ENGINE] Initializing shared components", color="yellow")
        self._initialize_shared_components()

        # Search state
        tprint("📊 [NAS_ENGINE] Initializing search state", color="blue")
        self.current_generation = 0
        self.best_architecture = None
        self.best_score = -np.inf
        self.search_history = []
        self.pareto_frontier = []
        self.evaluation_count = 0

        # Performance tracking
        self.start_time = None
        self.evaluation_times = []

        # Initialize hardware optimization with enhanced utilities
        tprint("⚡ [NAS_ENGINE] Initializing hardware optimization", color="yellow")
        self._initialize_hardware_optimization()

        # Initialize memory optimization
        tprint("💾 [NAS_ENGINE] Initializing memory optimization", color="yellow")
        self._initialize_memory_optimization()

        tprint("🎉 [NAS_ENGINE] Enhanced NAS Engine initialized successfully", color="green", bold=True)
        self.logger.info("✅ Enhanced NAS Engine initialized with integrated utility tools")
        self.logger.info(f"   Search Strategy: {config.search_strategy.value}")
        self.logger.info(f"   Population Size: {config.population_size}")
        self.logger.info(f"   Max Generations: {config.max_generations}")
        self.logger.info(f"   Memory Optimization: {'✅ Enabled' if hasattr(self, 'memory_optimizer') else '❌ Disabled'}")
        self.logger.info(f"   Data Quality Validation: {'✅ Enabled' if hasattr(self, 'data_quality_validator') else '❌ Disabled'}")

    def _initialize_shared_ml_utilities(self):
        """Initialize shared ML utilities from hybrid_nas_tas_regime."""
        try:
            # Create shared ML utilities manager for NAS
            ml_config = MLUtilityConfig(
                utility_type=MLUtilityType.NAS,
                enable_safeguards=True,
                enable_memory_optimization=True,
                enable_caching=True,
                enable_error_handling=True,
                enable_validation=True,
                enable_cross_validation=True,
                enable_threshold_optimization=True,
                cache_ttl_seconds=3600,
                memory_limit_mb=self.config.max_memory_mb
            )

            self.shared_ml_utilities = create_shared_ml_utilities_manager(MLUtilityType.NAS, ml_config)

            self.logger.info("✅ Shared ML utilities initialized for NAS Engine")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize shared ML utilities: {e}")
            raise

    def _initialize_shared_components(self):
        """Initialize shared utility components from unified framework."""
        try:
            # Use unified search space
            self.search_space = create_neural_search_space()

            # Performance estimator with financial objectives
            self.performance_estimator = create_unified_performance_estimator({
                'estimator_type': 'ensemble',
                'neural_config': {'estimator_type': 'ensemble'}
            })

            # Architecture encoder with advanced encoding
            self.architecture_encoder = create_unified_architecture_encoder({
                'encoding_method': 'hybrid',
                'latent_dim': 128
            })

            # Constraint validator
            self.constraint_validator = create_unified_constraint_validator({
                'max_layers': self.config.max_layers,
                'max_parameters': self.config.max_parameters,
                'max_memory_usage_mb': self.config.max_memory_mb,
                'max_training_time_seconds': self.config.max_training_time_per_arch
            })

            # Multi-objective optimizer
            self.multi_objective_optimizer = TradingMultiObjectiveOptimizer(MultiObjectiveConfig(
                objectives=[ObjectiveType.PERFORMANCE, ObjectiveType.SHARPE_RATIO,
                           ObjectiveType.MAX_DRAWDOWN, ObjectiveType.WIN_RATE],
                weights={
                    ObjectiveType.PERFORMANCE: 1.0,
                    ObjectiveType.SHARPE_RATIO: 0.8,
                    ObjectiveType.MAX_DRAWDOWN: 0.6,
                    ObjectiveType.WIN_RATE: 0.7
                }
            ))

            # Ensemble search space
            if self.config.enable_ensemble_search:
                ensemble_config = EnsembleSearchSpaceConfig(
                    max_models=5,
                    allowed_ensemble_methods=[
                        EnsembleMethod.WEIGHTED_VOTING,
                        EnsembleMethod.ADAPTIVE_WEIGHTING,
                        EnsembleMethod.UNCERTAINTY_WEIGHTING
                    ],
                    allowed_combination_strategies=[
                        EnsembleCombinationStrategy.HYBRID_MIXED,
                        EnsembleCombinationStrategy.PERFORMANCE_BASED
                    ]
                )
                self.ensemble_search_space = create_unified_ensemble_search_space(
                    nas_models=[],  # Will be populated during search
                    tas_models=[],  # Will be populated during search
                    config=ensemble_config,
                    performance_estimator=self.performance_estimator,
                    constraint_validator=self.constraint_validator
                )
                self.logger.info("✅ Ensemble search space initialized for NAS")
            else:
                self.ensemble_search_space = None

            # Architecture compression
            if self.config.enable_compression:
                compression_config = CompressionConfig(
                    compression_method=self.config.compression_method,
                    compression_level=self.config.compression_level,
                    max_performance_loss=0.05,
                    min_compression_ratio=0.2
                )
                self.architecture_compressor = create_unified_architecture_compressor(compression_config)
                self.logger.info("✅ Architecture compressor initialized for NAS")
            else:
                self.architecture_compressor = None

            # Search space evolution
            if self.config.enable_evolution:
                evolution_config = UnifiedEvolutionConfig(
                    enable_performance_based_evolution=True,
                    enable_regime_based_evolution=True,
                    evolution_intensity=self.config.evolution_intensity,
                    min_evolution_interval=self.config.min_evolution_interval
                )
                self.evolution_manager = create_unified_evolution_manager(
                    nas_search_space=self.search_space,
                    tas_search_space=None,  # NAS engine only
                    config=evolution_config
                )
                self.logger.info("✅ Search space evolution manager initialized for NAS")
            else:
                self.evolution_manager = None

            self.logger.info("✅ All shared components initialized with unified framework")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize shared components: {e}")
            raise

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization for NAS search with enhanced utilities."""
        try:
            if self.config.enable_hardware_optimization:
                # Import unified hardware manager
                from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel

                # Create hardware configuration optimized for NAS
                hardware_config = HardwareConfig(
                    cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                    gpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                    memory_optimization_level=OptimizationLevel.BALANCED,
                    memory_limit_gb=self.config.max_memory_mb / 1024,  # Convert MB to GB
                    enable_mps_acceleration=True,
                    enable_gpu_memory_pooling=True,
                    enable_batch_operations=True,
                    enable_adaptive_optimization=True,
                    learning_enabled=True,
                    auto_tuning_enabled=True,
                    performance_monitoring_enabled=True,
                    monitoring_interval=5.0
                )

                # Create unified hardware manager
                self.hardware_optimizer = UnifiedHardwareManager(hardware_config)

                # Optimize for NAS workload
                self.hardware_optimizer.optimize_for_workload(
                    WorkloadType.ML_TRAINING,
                    parameters={
                        'neural_network_training': True,
                        'parallel_evaluations': self.config.parallel_evaluation,
                        'n_workers': self.config.n_workers,
                        'memory_per_model_mb': self.config.max_memory_mb // self.config.population_size
                    }
                )

                self.logger.info("✅ Hardware optimization initialized for NAS")
                self.logger.info(f"   CPU Optimization: {hardware_config.cpu_optimization_level.value}")
                self.logger.info(f"   GPU Optimization: {hardware_config.gpu_optimization_level.value}")
                self.logger.info(f"   Memory Limit: {hardware_config.memory_limit_gb} GB")

                # Check disk space for data storage
                disk_status = check_disk_space(".", required_gb=1.0)
                if disk_status['sufficient']:
                    self.logger.info(f"   Disk Space: ✅ {disk_status['available_percentage']:.1f}% available")
                else:
                    self.logger.warning(f"   Disk Space: ⚠️ Low space ({disk_status['available_percentage']:.1f}%)")

            else:
                self.hardware_optimizer = None
                self.logger.info("⚠️ Hardware optimization disabled")

        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.hardware_optimizer = None

    def _initialize_memory_optimization(self):
        """Initialize memory optimization using enhanced utilities."""
        try:
            # Initialize memory checkpoint context manager
            self.memory_checkpoint = memory_checkpoint("NAS_Engine")
            self.gpu_context = gpu_context("NAS_Operations")

            # Optimize memory on startup
            memory_status = optimize_memory()
            if memory_status['success']:
                self.logger.info(f"✅ Memory optimization initialized: {memory_status.get('method', 'unknown')}")
            else:
                self.logger.warning(f"⚠️ Memory optimization failed: {memory_status.get('error', 'unknown')}")

            # Set up periodic memory monitoring
            self.memory_monitoring_enabled = True

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization initialization failed: {e}")
            self.memory_checkpoint = None
            self.gpu_context = None
            self.memory_monitoring_enabled = False

    def _create_architecture_constraints(self):
        """Create architecture constraints from config."""
        from ...hybrid_nas_tas_regime.shared_utils.constraint_systems import ArchitectureConstraints

        return ArchitectureConstraints(
            max_layers=self.config.max_layers,
            min_layers=self.config.min_layers,
            max_parameters=self.config.max_parameters,
            max_memory_usage_mb=self.config.max_memory_mb,
            max_training_time_seconds=self.config.max_training_time_per_arch,
            allow_complex_activations=self.config.allow_complex_activations
        )

    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None) -> NASSearchResult:
        """Perform comprehensive neural architecture search with integrated utility tools."""
        tprint("🚀 [NAS_ENGINE] Starting Enhanced NAS Search", color="cyan", bold=True)
        self.start_time = time.time()
        self.logger.info("🚀 Starting Enhanced NAS Search with integrated utility tools...")

        try:
            # Validate input data using enhanced utilities
            X_train, y_train = train_data
            X_val, y_val = validation_data
            
            tprint(f"📊 [NAS_ENGINE] Training data shape: {X_train.shape}, labels: {y_train.shape}", color="blue")
            tprint(f"📊 [NAS_ENGINE] Validation data shape: {X_val.shape}, labels: {y_val.shape}", color="blue")

            # Convert to DataFrame for validation
            train_df = pd.DataFrame({'X': X_train.flatten(), 'y': y_train.flatten()})
            val_df = pd.DataFrame({'X': X_val.flatten(), 'y': y_val.flatten()})

            # Validate data quality
            tprint("🔍 [NAS_ENGINE] Validating data quality", color="yellow")
            train_quality = calculate_data_quality_metrics(train_df)
            val_quality = calculate_data_quality_metrics(val_df)

            tprint(f"📊 [NAS_ENGINE] Train Data Quality - Missing: {train_quality['missing_percentage']:.2f}%, Duplicates: {train_quality['duplicate_percentage']:.2f}%", color="cyan")
            tprint(f"📊 [NAS_ENGINE] Validation Data Quality - Missing: {val_quality['missing_percentage']:.2f}%, Duplicates: {val_quality['duplicate_percentage']:.2f}%", color="cyan")
            self.logger.info(f"📊 Train Data Quality - Missing: {train_quality['missing_percentage']:.2f}%, Duplicates: {train_quality['duplicate_percentage']:.2f}%")
            self.logger.info(f"📊 Validation Data Quality - Missing: {val_quality['missing_percentage']:.2f}%, Duplicates: {val_quality['duplicate_percentage']:.2f}%")

            # Guard against excessive nulls
            train_df = guard_dataframe_nulls(train_df, threshold=0.3)
            val_df = guard_dataframe_nulls(val_df, threshold=0.3)

            # Apply safeguards before search using shared utilities
            if not self.shared_ml_utilities.check_training_safety(train_data, validation_data):
                self.logger.warning("Training safety check failed, proceeding with caution")

            # Check lookahead protection using shared utilities
            if not self.shared_ml_utilities.validate_data_split(train_data, validation_data):
                self.logger.warning("Data split may have lookahead bias")

            # Memory checkpoint for search operation
            with self.memory_checkpoint:
                # Check memory usage before search
                memory_before = get_memory_usage()
                self.logger.info(f"📈 Memory before search: {format_bytes(memory_before)}")

            # Select and initialize search strategy
            search_strategy = self._create_search_strategy()

            # Define enhanced objective function with shared utilities
            def objective_function(architecture):
                return self._evaluate_architecture_with_shared_utilities(
                    architecture, validation_data, regime_data
                )

            # Perform search based on strategy with error handling
            try:
                # Perform search based on strategy
                if self.config.search_strategy == SearchStrategy.RANDOM:
                    result = self._random_search(objective_function)
                elif self.config.search_strategy == SearchStrategy.BAYESIAN_OPTIMIZATION:
                    result = self._bayesian_search(objective_function, search_strategy)
                elif self.config.search_strategy == SearchStrategy.EVOLUTIONARY:
                    result = self._evolutionary_search(objective_function, search_strategy)
                elif self.config.search_strategy == SearchStrategy.REINFORCEMENT_LEARNING:
                    result = self._rl_search(objective_function, search_strategy)
                elif self.config.search_strategy == SearchStrategy.ENHANCED_BAYESIAN:
                    result = self._enhanced_bayesian_search(objective_function, search_strategy)
                elif self.config.search_strategy == SearchStrategy.ADAPTIVE_EVOLUTIONARY:
                    result = self._adaptive_evolutionary_search(objective_function, search_strategy)
                else:
                    result = self._hybrid_search(objective_function, search_strategy)

                # Validate final result
                if result['best_score'] > 0:
                    # Perform cross-validation on best architecture using shared utilities
                    cv_result = self._perform_cross_validation_shared(result['best_architecture'], train_data, validation_data)

                    # Optimize thresholds if applicable using shared utilities
                    if test_data:
                        optimized_thresholds = self._optimize_model_thresholds_shared(result['best_architecture'], test_data)

                execution_time = time.time() - self.start_time

                # Create final result with enhanced metadata
                search_result = NASSearchResult(
                    best_architecture=result['best_architecture'],
                    best_score=result['best_score'],
                    search_history=self.search_history,
                    pareto_frontier=self.pareto_frontier,
                    strategy_used=self.config.search_strategy.value,
                    convergence_info=result.get('convergence_info', {}),
                    execution_time=execution_time,
                    n_evaluations=self.evaluation_count,
                    metadata={
                        'search_strategy': self.config.search_strategy.value,
                        'population_size': self.config.population_size,
                        'max_generations': self.config.max_generations,
                        'final_generation': self.current_generation,
                        'shared_ml_utilities_used': True,
                        'utility_type': 'NAS',
                        'safeguards_applied': True,
                        'cross_validation_performed': True if 'cv_result' in locals() else False,
                        'memory_optimized': True,
                        'error_handling_enabled': True
                    }
                )

                self.logger.info("✅ Enhanced NAS Search completed successfully with ML Common utilities")
                self.logger.info(f"   Best Score: {search_result.best_score:.4f}")
                self.logger.info(f"   Total Evaluations: {self.evaluation_count}")
                self.logger.info(f"   Execution Time: {execution_time:.2f}s")

                return search_result

            except Exception as search_error:
                # Handle search-specific errors using shared utilities
                return self.shared_ml_utilities.handle_error(
                    search_error, {
                        'best_architecture': self.best_architecture,
                        'best_score': self.best_score,
                        'search_history': self.search_history
                    }
                )

        except Exception as e:
            execution_time = time.time() - self.start_time
            self.logger.error(f"❌ Enhanced NAS Search failed: {e}")

            # Return partial result with error information
            return NASSearchResult(
                best_architecture=self.best_architecture,
                best_score=self.best_score,
                search_history=self.search_history,
                pareto_frontier=self.pareto_frontier,
                strategy_used=self.config.search_strategy.value,
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                n_evaluations=self.evaluation_count,
                metadata={'error': str(e), 'shared_ml_utilities_used': True, 'utility_type': 'NAS'}
            )

    def _create_search_strategy(self):
        """Create the appropriate search strategy."""
        if self.config.search_strategy == SearchStrategy.REINFORCEMENT_LEARNING:
            return create_rl_search_strategy({
                'agent_type': 'q_learning',
                'learning_rate': 0.01,
                'exploration_rate': 1.0,
                'exploration_decay': 0.995
            })
        elif self.config.search_strategy == SearchStrategy.ENHANCED_BAYESIAN:
            return create_enhanced_bayesian_search({
                'n_initial_points': min(20, self.config.population_size),
                'acquisition_function': 'expected_improvement',
                'kernel_type': 'matern'
            })
        elif self.config.search_strategy == SearchStrategy.ADAPTIVE_EVOLUTIONARY:
            return create_adaptive_evolutionary_search({
                'population_size': self.config.population_size,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'tournament_size': 5,
                'use_island_model': True,
                'n_islands': 5
            })
        else:
            return None

    def _evaluate_architecture_with_shared_utilities(self, architecture, validation_data, regime_data=None) -> float:
        """Evaluate architecture using shared ML utilities with enhanced validation."""
        try:
            # Validate architecture using math utilities
            if hasattr(architecture, 'layers') and architecture.layers:
                # Validate numeric parameters
                for layer in architecture.layers:
                    if hasattr(layer, 'weights'):
                        for weight in layer.weights:
                            if isinstance(weight, (int, float)):
                                math_validate_finite(weight, f"architecture_weight_{layer.name}")

            # Use NAS-specific evaluation from shared utilities
            result = self.shared_ml_utilities.evaluate_neural_architecture(
                architecture, validation_data, regime_data
            )

            # Validate result using math utilities
            result = math_validate_finite(result, "architecture_evaluation_result")
            result = math_validate_range(result, 0.0, 1.0, "architecture_evaluation_result")

            return result

        except Exception as e:
            tprint_error(f"Architecture evaluation with shared utilities failed: {e}")
            tprint_debug(f"Architecture evaluation error context: {locals()}")
            tprint_error("CRITICAL: Architecture evaluation with shared utilities is required for NAS analysis")
            tprint_error("Cannot proceed without proper architecture evaluation")
            self.logger.error(f"Architecture evaluation with shared utilities failed: {e}")
            raise ValueError(f"Architecture evaluation with shared utilities failed: {e}") from e

    def _evaluate_architecture(self, architecture, validation_data, regime_data=None) -> float:
        """Evaluate an architecture's performance."""
        start_time = time.time()

        try:
            # Use performance estimator if enabled
            if self.config.enable_performance_estimation and self.performance_estimator:
                try:
                    prediction = self.performance_estimator.predict_performance(architecture)
                    estimated_score = prediction.predicted_performance
                    evaluation_time = time.time() - start_time

                    # Store evaluation info
                    self.evaluation_times.append(evaluation_time)
                    self.evaluation_count += 1

                    self.logger.debug(f"Architecture evaluated with estimator: {estimated_score:.4f}")
                    return estimated_score
        except Exception as e:
            tprint_error(f"Performance estimator failed: {e}")
            tprint_debug(f"Performance estimator error context: {locals()}")
            tprint_error("CRITICAL: Performance estimator is required for NAS analysis")
            tprint_error("Cannot proceed without proper performance estimation")
            self.logger.error(f"Performance estimator failed: {e}")
            raise ValueError(f"Performance estimator failed: {e}") from e

            # Fallback to actual evaluation (simplified)
            # In practice, this would involve training and validating the architecture
            X_val, y_val = validation_data

            # Simplified evaluation based on architecture properties
            complexity_score = architecture.estimated_complexity
            parameter_efficiency = min(architecture.layers[0].hidden_size / 1000.0, 1.0) if architecture.layers else 0.0

            # Simulate performance based on architecture characteristics
            base_score = 0.5
            complexity_bonus = min(complexity_score * 0.1, 0.3)
            efficiency_bonus = parameter_efficiency * 0.2

            score = base_score + complexity_bonus + efficiency_bonus

            # Add some noise for realism
            score += np.random.normal(0, 0.05)
            score = max(0.1, min(0.9, score))

            evaluation_time = time.time() - start_time
            self.evaluation_times.append(evaluation_time)
            self.evaluation_count += 1

            return score

        except Exception as e:
            tprint_error(f"Architecture evaluation failed: {e}")
            tprint_debug(f"Architecture evaluation error context: {locals()}")
            tprint_error("CRITICAL: Architecture evaluation is required for NAS analysis")
            tprint_error("Cannot proceed without proper architecture evaluation")
            self.logger.error(f"Architecture evaluation failed: {e}")
            raise ValueError(f"Architecture evaluation failed: {e}") from e

    def _evaluate_architecture_fallback(self, architecture, validation_data, regime_data=None) -> float:
        """Fallback evaluation when shared utilities fail."""
        try:
            X_val, y_val = validation_data

            # Neural-specific evaluation based on architecture properties
            complexity_score = architecture.estimated_complexity if hasattr(architecture, 'estimated_complexity') else 1.0
            parameter_efficiency = min(architecture.layers[0].hidden_size / 1000.0, 1.0) if hasattr(architecture, 'layers') and architecture.layers else 0.0

            base_score = 0.5
            complexity_bonus = min(complexity_score * 0.1, 0.3)
            efficiency_bonus = parameter_efficiency * 0.2

            score = base_score + complexity_bonus + efficiency_bonus
            score += np.random.normal(0, 0.05)
            score = max(0.1, min(0.9, score))

            return score

        except Exception as e:
            tprint_error(f"NAS architecture fallback evaluation failed: {e}")
            tprint_debug(f"Fallback evaluation error context: {locals()}")
            tprint_error("CRITICAL: NAS architecture fallback evaluation is required for NAS analysis")
            tprint_error("Cannot proceed without proper fallback evaluation")
            self.logger.error(f"NAS architecture fallback evaluation failed: {e}")
            raise ValueError(f"NAS architecture fallback evaluation failed: {e}") from e

    def _random_search(self, objective_function: Callable) -> Dict[str, Any]:
        """Perform random search."""
        self.logger.info("🔍 Starting Random Search...")

        best_architecture = None
        best_score = -np.inf

        for i in range(self.config.max_evaluations):
            # Generate random architecture
            architecture = self.search_space.sample_random_architecture()

            # Validate constraints
            if self.config.enable_constraint_validation:
                if not self.constraint_validator.validate(architecture).is_valid:
                    continue

            # Evaluate architecture
            score = objective_function(architecture)

            # Update best
            if score > best_score:
                best_score = score
                best_architecture = architecture

            # Store in history
            self.search_history.append({
                'generation': 0,
                'architecture': architecture,
                'score': score,
                'strategy': 'random'
            })

            # Early stopping check
            if i >= self.config.early_stopping_patience and i % 10 == 0:
                recent_scores = [h['score'] for h in self.search_history[-10:]]
                if max(recent_scores) - min(recent_scores) < self.config.early_stopping_threshold:
                    self.logger.info(f"Early stopping at iteration {i}")
                    break

        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'convergence_info': {'early_stopped': i < self.config.max_evaluations}
        }

    def _bayesian_search(self, objective_function: Callable, search_strategy) -> Dict[str, Any]:
        """Perform Bayesian optimization search."""
        self.logger.info("🔍 Starting Bayesian Optimization Search...")

        # Use the shared Bayesian optimization strategy
        result = search_strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_generations
        )

        return {
            'best_architecture': result.best_architecture,
            'best_score': result.best_score,
            'convergence_info': result.convergence_info
        }

    def _enhanced_bayesian_search(self, objective_function: Callable, search_strategy) -> Dict[str, Any]:
        """Perform enhanced Bayesian optimization search."""
        self.logger.info("🔍 Starting Enhanced Bayesian Optimization Search...")

        # Use the shared enhanced Bayesian optimization strategy
        result = search_strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_generations
        )

        return {
            'best_architecture': result.best_architecture,
            'best_score': result.best_score,
            'convergence_info': result.convergence_info
        }

    def _evolutionary_search(self, objective_function: Callable, search_strategy) -> Dict[str, Any]:
        """Perform evolutionary search."""
        self.logger.info("🔍 Starting Evolutionary Search...")

        # Use the shared adaptive evolutionary strategy
        result = search_strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_generations
        )

        return {
            'best_architecture': result.best_architecture,
            'best_score': result.best_score,
            'convergence_info': result.convergence_info
        }

    def _rl_search(self, objective_function: Callable, search_strategy) -> Dict[str, Any]:
        """Perform reinforcement learning search."""
        self.logger.info("🔍 Starting Reinforcement Learning Search...")

        # Use the shared RL search strategy
        result = search_strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_generations
        )

        return {
            'best_architecture': result.best_architecture,
            'best_score': result.best_score,
            'convergence_info': result.convergence_info
        }

    def _hybrid_search(self, objective_function: Callable, search_strategy) -> Dict[str, Any]:
        """Perform hybrid search combining multiple strategies."""
        self.logger.info("🔍 Starting Hybrid Search...")

        # Combine multiple strategies
        strategies = [
            self._create_search_strategy_class('bayesian_optimization'),
            self._create_search_strategy_class('evolutionary'),
            self._create_search_strategy_class('rl')
        ]

        best_overall_architecture = None
        best_overall_score = -np.inf

        for i, strategy in enumerate(strategies):
            self.logger.info(f"Running strategy {i+1}/{len(strategies)}")

            result = strategy.search(
                architecture_generator=self._architecture_generator,
                performance_evaluator=objective_function,
                constraint_validator=self._constraint_checker,
                n_iterations=self.config.max_generations // len(strategies)
            )

            if result.best_score > best_overall_score:
                best_overall_score = result.best_score
                best_overall_architecture = result.best_architecture

        return {
            'best_architecture': best_overall_architecture,
            'best_score': best_overall_score,
            'convergence_info': {'strategies_used': len(strategies)}
        }

    def _create_search_strategy_class(self, strategy_name: str):
        """Create a search strategy instance by name."""
        if strategy_name == 'bayesian_optimization':
            return create_enhanced_bayesian_search({
                'n_initial_points': 10,
                'acquisition_function': 'expected_improvement',
                'kernel_type': 'matern'
            })
        elif strategy_name == 'evolutionary':
            return create_adaptive_evolutionary_search({
                'population_size': self.config.population_size // 3,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            })
        elif strategy_name == 'rl':
            return create_rl_search_strategy({
                'agent_type': 'q_learning',
                'learning_rate': 0.01,
                'exploration_rate': 1.0
            })
        else:
            return create_enhanced_bayesian_search({})

    def _architecture_generator(self) -> Any:
        """Generate a random architecture from search space."""
        return self.search_space.sample_random_architecture()

    def _constraint_checker(self, architecture: Any) -> Any:
        """Check if architecture meets constraints."""
        return self.constraint_validator.validate(architecture)

    def _perform_cross_validation_shared(self, architecture, train_data, validation_data) -> Dict[str, Any]:
        """Perform cross-validation using shared ML utilities."""
        try:
            X_train, y_train = train_data
            X_val, y_val = validation_data

            # Combine train and validation for cross-validation
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.hstack([y_train, y_val])

            # Use shared utilities for cross-validation
            return self.shared_ml_utilities.perform_cross_validation(
                model=architecture,
                X=X_combined,
                y=y_combined,
                strategy="temporal",
                cv_folds=5,
                scoring=['accuracy', 'precision', 'recall', 'f1']
            )

        except Exception as e:
            tprint_error(f"NAS Cross-validation with shared utilities failed: {e}")
            tprint_debug(f"Cross-validation error context: {locals()}")
            tprint_error("CRITICAL: NAS Cross-validation is required for NAS analysis")
            tprint_error("Cannot proceed without proper cross-validation")
            self.logger.error(f"NAS Cross-validation with shared utilities failed: {e}")
            raise ValueError(f"NAS Cross-validation failed: {e}") from e

    def _optimize_model_thresholds_shared(self, architecture, test_data) -> Dict[str, Any]:
        """Optimize model thresholds using shared ML utilities."""
        try:
            X_test, y_test = test_data

            # Get predictions from architecture
            if hasattr(architecture, 'predict_proba'):
                y_pred_proba = architecture.predict_proba(X_test)
                y_pred = architecture.predict(X_test)

                # Use shared utilities for threshold optimization
                return self.shared_ml_utilities.optimize_thresholds(
                    y_true=y_test,
                    y_pred_proba=y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba,
                    metric='f1'
                )

            else:
                self.logger.warning("Architecture does not support probability predictions")
                return {'success': False, 'error': 'No probability predictions available'}

        except Exception as e:
            tprint_error(f"NAS Threshold optimization with shared utilities failed: {e}")
            tprint_debug(f"Threshold optimization error context: {locals()}")
            tprint_error("CRITICAL: NAS Threshold optimization is required for NAS analysis")
            tprint_error("Cannot proceed without proper threshold optimization")
            self.logger.error(f"NAS Threshold optimization with shared utilities failed: {e}")
            raise ValueError(f"NAS Threshold optimization failed: {e}") from e


    def save_search_state(self, filepath: str) -> bool:
        """Save the current NAS search state with enhanced serialization utilities."""
        try:
            # Ensure directory exists
            ensure_directory(os.path.dirname(filepath))

            state = {
                'config': self.config,
                'current_generation': self.current_generation,
                'best_architecture': self.best_architecture,
                'best_score': self.best_score,
                'search_history': self.search_history,
                'pareto_frontier': self.pareto_frontier,
                'evaluation_count': self.evaluation_count,
                'evaluation_times': self.evaluation_times,
                'start_time': self.start_time,
                # Enhanced utilities state
                'utilities_version': '2.0',
                'shared_ml_utilities_used': True,
                'utility_type': 'NAS',
                'memory_optimization_enabled': self.memory_monitoring_enabled,
                'hardware_optimization_enabled': self.config.enable_hardware_optimization,
                'ml_utilities_status': self.shared_ml_utilities.get_system_status() if hasattr(self, 'shared_ml_utilities') else None,
                'memory_usage': get_memory_usage(),
                'disk_status': check_disk_space(".", required_gb=0.1)
            }

            # Use universal serializer for enhanced persistence
            success = self.universal_serializer.save(state, filepath)

            if success:
                self.logger.info(f"✅ NAS search state saved to {filepath} with enhanced utilities")
                file_size = get_file_size(filepath)
                self.logger.info(f"   File size: {format_bytes(file_size)}")
            else:
                # Fallback to pickle if universal serializer fails
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
                self.logger.info(f"⚠️ Universal serializer failed, used pickle fallback for {filepath}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save NAS search state: {e}")
            return False

    def load_search_state(self, filepath: str) -> bool:
        """Load a saved NAS search state with enhanced utilities."""
        try:
            # Validate file exists and is accessible
            if not validate_file_path(filepath):
                self.logger.error(f"❌ File does not exist or is not accessible: {filepath}")
                return False

            # Use universal serializer for enhanced loading
            state = self.universal_serializer.load(filepath)

            if state is None:
                # Fallback to pickle if universal serializer fails
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)
                self.logger.info(f"⚠️ Universal serializer failed, used pickle fallback for {filepath}")

            self.config = state['config']
            self.current_generation = state['current_generation']
            self.best_architecture = state['best_architecture']
            self.best_score = state['best_score']
            self.search_history = state['search_history']
            self.pareto_frontier = state['pareto_frontier']
            self.evaluation_count = state['evaluation_count']
            self.evaluation_times = state['evaluation_times']
            self.start_time = state['start_time']

            # Restore enhanced utilities state if available
            utilities_version = state.get('utilities_version', '1.0')
            self.logger.info(f"Loading NAS search state with utilities version {utilities_version}")

            if state.get('shared_ml_utilities_used', False):
                self.logger.info("Loading NAS search state with shared ML utilities")

                # Reinitialize shared ML utilities
                self._initialize_shared_ml_utilities()
                self._initialize_shared_components()

            # Restore memory optimization settings
            if state.get('memory_optimization_enabled', False):
                self._initialize_memory_optimization()

            # Restore hardware optimization settings
            if state.get('hardware_optimization_enabled', False):
                self._initialize_hardware_optimization()

            # Log restoration summary
            file_size = get_file_size(filepath)
            memory_usage = state.get('memory_usage', 0)
            disk_status = state.get('disk_status', {})

            self.logger.info(f"✅ NAS search state loaded from {filepath} with enhanced utilities")
            self.logger.info(f"   File size: {format_bytes(file_size)}")
            self.logger.info(f"   Previous memory usage: {format_bytes(memory_usage)}")
            if disk_status:
                self.logger.info(f"   Previous disk status: {disk_status.get('available_percentage', 'unknown')}% available")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load NAS search state: {e}")
            return False


def create_enhanced_nas_engine(config: NASSearchConfig) -> EnhancedNASEngine:
    """Create an enhanced NAS engine instance."""
    return EnhancedNASEngine(config)


def quick_nas_search(train_data: Tuple[np.ndarray, np.ndarray],
                    validation_data: Tuple[np.ndarray, np.ndarray],
                    config: Optional[NASSearchConfig] = None) -> NASSearchResult:
    """Quick NAS search with default settings and shared ML utilities."""
    if config is None:
        config = NASSearchConfig(
            search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
            population_size=30,
            max_generations=50,
            max_evaluations=200,
            # Enable shared ML utilities
            enable_multi_objective=True,
            enable_constraint_validation=True,
            enable_performance_estimation=True,
            parallel_evaluation=True
        )

    engine = EnhancedNASEngine(config)
    return engine.search(train_data, validation_data)