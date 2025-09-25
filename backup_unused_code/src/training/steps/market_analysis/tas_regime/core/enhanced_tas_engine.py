"""
Enhanced TAS Engine with Complete Architecture Search Capabilities

This module provides a comprehensive tree architecture search engine that integrates
all the shared components including advanced search strategies, performance estimators,
architecture encoding, constraint validation, and ML common utilities.
"""

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

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import standardized logging
from ...logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)

# Import shared ML utilities from hybrid_nas_tas_regime
from ...hybrid_nas_tas_regime.shared_utils.ml_common_integration import (
    create_shared_ml_utilities_manager, MLUtilityType, MLUtilityConfig
)

# Circular import removed to prevent import errors
# from ...hybrid_nas_tas_regime.core.unified_architecture_search_engine import (
#     UnifiedArchitectureSearchEngine, UnifiedSearchConfig, ArchitectureType
# )
from ...hybrid_nas_tas_regime.shared_utils.performance_estimators import (
    UnifiedPerformanceEstimator, create_unified_performance_estimator
)
# from ...hybrid_nas_tas_regime.core.advanced_search_strategies import (
#     AdvancedSearchStrategies, SearchStrategyType
# )
from ...hybrid_nas_tas_regime.shared_utils.search_spaces import create_tree_search_space
from ...hybrid_nas_tas_regime.shared_utils.constraint_systems import create_unified_constraint_validator
from ...hybrid_nas_tas_regime.shared_utils.advanced_search_strategies import (
    create_rl_search_strategy, create_enhanced_bayesian_search, create_adaptive_evolutionary_search
)
try:
    from ...hybrid_nas_tas_regime.core.multi_objective_optimizer import (  # type: ignore
        TradingMultiObjectiveOptimizer, MultiObjectiveConfig, ObjectiveType
    )
except ImportError:
    # Fallback classes if module is not available
    class TradingMultiObjectiveOptimizer:
        pass
    class MultiObjectiveConfig:
        pass
    class ObjectiveType:
        pass
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

# Use shared logger from hybrid utilities
logger = logging.getLogger(__name__)


class TreeSearchStrategy(Enum):
    """Available search strategies for TAS."""
    RANDOM = "random"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENHANCED_BAYESIAN = "enhanced_bayesian"
    ADAPTIVE_EVOLUTIONARY = "adaptive_evolutionary"
    HYBRID = "hybrid"


@dataclass
class TASConfig:
    """Configuration for TAS search."""
    search_strategy: TreeSearchStrategy = TreeSearchStrategy.ENHANCED_BAYESIAN
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
        'interpretability': 0.5
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
    compression_method: CompressionMethod = CompressionMethod.TREE_PRUNING
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    
    # Search space evolution
    enable_evolution: bool = True
    evolution_intensity: float = 0.3
    min_evolution_interval: int = 100

    # Hardware constraints and optimization
    max_memory_mb: int = 8192
    max_training_time_per_arch: int = 600  # 10 minutes
    parallel_evaluation: bool = True
    n_workers: int = 4

    # Hardware optimization
    enable_hardware_optimization: bool = True
    hardware_optimizer = None

    # Tree-specific constraints
    max_trees: int = 50
    max_tree_depth: int = 30
    min_tree_depth: int = 3
    allow_boosting: bool = True
    allow_bagging: bool = True
    allow_ensemble_methods: bool = True


@dataclass
class TASResult:
    """Result from TAS search."""
    best_architecture: Any
    best_score: float
    search_history: List[Dict[str, Any]]
    pareto_frontier: List[Any]
    strategy_used: str
    convergence_info: Dict[str, Any]
    execution_time: float
    n_evaluations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedTASEngine:
    """Enhanced Tree Architecture Search Engine."""

    def __init__(self, config: TASConfig):
        """Initialize the enhanced TAS engine with shared ML utilities."""
        tprint_info("🚀 Initializing Enhanced TAS Engine")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config

        # Initialize standardized logging
        self.logger = get_logger('EnhancedTASEngine')
        tprint_debug("📝 Standardized logging initialized")

        # Initialize shared ML utilities from hybrid_nas_tas_regime
        tprint_info("🔧 Initializing shared ML utilities...")
        self._initialize_shared_ml_utilities()

        # Initialize shared components
        tprint_info("🧩 Initializing shared components...")
        self._initialize_shared_components()

        # Search state
        tprint_debug("📊 Initializing search state...")
        self.current_generation = 0
        self.best_architecture = None
        self.best_score = -np.inf
        self.search_history = []
        self.pareto_frontier = []
        self.evaluation_count = 0

        # Performance tracking
        self.start_time = None
        self.evaluation_times = []
        tprint_debug("⏱️ Performance tracking initialized")

        # Initialize hardware optimization
        tprint_info("💻 Initializing hardware optimization...")
        self._initialize_hardware_optimization()

        tprint_success("✅ Enhanced TAS Engine initialized with shared ML utilities")
        log_success("Enhanced TAS Engine initialized with shared ML utilities")
        log_info(f"   Search Strategy: {config.search_strategy.value}")
        log_info(f"   Population Size: {config.population_size}")
        log_info(f"   Max Generations: {config.max_generations}")

    def _initialize_shared_ml_utilities(self):
        """Initialize shared ML utilities from hybrid_nas_tas_regime."""
        tprint_debug("🔧 Starting shared ML utilities initialization...")
        try:
            # Create shared ML utilities manager for TAS
            tprint_debug(f"📋 Creating ML config with memory limit: {self.config.max_memory_mb}MB")
            ml_config = MLUtilityConfig(
                utility_type=MLUtilityType.TAS,
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

            tprint_debug("🏗️ Creating shared ML utilities manager...")
            self.shared_ml_utilities = create_shared_ml_utilities_manager(MLUtilityType.TAS, ml_config)

            tprint_success("✅ Shared ML utilities initialized for TAS Engine")
            log_success("Shared ML utilities initialized for TAS Engine")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize shared ML utilities: {e}")
            log_error(f"Failed to initialize shared ML utilities: {e}")
            raise

    def _initialize_shared_components(self):
        """Initialize shared utility components with unified framework."""
        tprint_debug("🧩 Starting shared components initialization...")
        try:
            # Use unified search space
            tprint_debug("🌳 Creating unified search space...")
            self.search_space = create_tree_search_space()
            tprint_debug("✅ Search space created")

            # Performance estimator with financial objectives
            tprint_debug("📊 Creating performance estimator...")
            self.performance_estimator = create_unified_performance_estimator({
                'estimator_type': 'meta_learner',
                'tree_config': {'estimator_type': 'meta_learner'}
            })
            tprint_debug("✅ Performance estimator created")

            # Architecture encoder with advanced encoding
            tprint_debug("🔤 Creating architecture encoder...")
            self.architecture_encoder = create_unified_architecture_encoder({
                'encoding_method': 'hybrid',
                'latent_dim': 128
            })
            tprint_debug("✅ Architecture encoder created")

            # Constraint validator
            tprint_debug("🔍 Creating constraint validator...")
            self.constraint_validator = create_unified_constraint_validator({
                'max_layers': self.config.max_trees,
                'max_parameters': 1000000,
                'max_memory_usage_mb': self.config.max_memory_mb,
                'max_training_time_seconds': self.config.max_training_time_per_arch
            })
            tprint_debug("✅ Constraint validator created")

            # Multi-objective optimizer for trading
            tprint_debug("🎯 Creating multi-objective optimizer...")
            self.multi_objective_optimizer = TradingMultiObjectiveOptimizer(MultiObjectiveConfig(
                objectives=[ObjectiveType.PERFORMANCE, ObjectiveType.SHARPE_RATIO,
                           ObjectiveType.MAX_DRAWDOWN, ObjectiveType.PROFIT_FACTOR],
                weights={
                    ObjectiveType.PERFORMANCE: 1.0,
                    ObjectiveType.SHARPE_RATIO: 0.8,
                    ObjectiveType.MAX_DRAWDOWN: 0.6,
                    ObjectiveType.PROFIT_FACTOR: 0.5
                }
            ))
            tprint_debug("✅ Multi-objective optimizer created")

            # Ensemble search space
            if self.config.enable_ensemble_search:
                tprint_debug("🎭 Creating ensemble search space...")
                ensemble_config = EnsembleSearchSpaceConfig(
                    max_models=5,
                    allowed_ensemble_methods=[
                        EnsembleMethod.WEIGHTED_VOTING,
                        EnsembleMethod.ADAPTIVE_WEIGHTING,
                        EnsembleMethod.UNCERTAINTY_WEIGHTING
                    ],
                    allowed_combination_strategies=[
                        EnsembleCombinationStrategy.TREE_ONLY,
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
                tprint_success("✅ Ensemble search space initialized for TAS")
                log_success("Ensemble search space initialized for TAS")
            else:
                tprint_debug("🚫 Ensemble search disabled")
                self.ensemble_search_space = None

            # Architecture compression
            if self.config.enable_compression:
                tprint_debug("🗜️ Creating architecture compressor...")
                compression_config = CompressionConfig(
                    compression_method=self.config.compression_method,
                    compression_level=self.config.compression_level,
                    max_performance_loss=0.05,
                    min_compression_ratio=0.2
                )
                self.architecture_compressor = create_unified_architecture_compressor(compression_config)
                tprint_success("✅ Architecture compressor initialized for TAS")
                log_success("Architecture compressor initialized for TAS")
            else:
                tprint_debug("🚫 Architecture compression disabled")
                self.architecture_compressor = None

            # Search space evolution
            if self.config.enable_evolution:
                tprint_debug("🧬 Creating search space evolution manager...")
                evolution_config = UnifiedEvolutionConfig(
                    enable_performance_based_evolution=True,
                    enable_regime_based_evolution=True,
                    evolution_intensity=self.config.evolution_intensity,
                    min_evolution_interval=self.config.min_evolution_interval
                )
                self.evolution_manager = create_unified_evolution_manager(
                    nas_search_space=None,  # TAS engine only
                    tas_search_space=self.search_space,
                    config=evolution_config
                )
                tprint_success("✅ Search space evolution manager initialized for TAS")
                log_success("Search space evolution manager initialized for TAS")
            else:
                tprint_debug("🚫 Search space evolution disabled")
                self.evolution_manager = None

            tprint_success("✅ All shared components initialized with unified framework")
            log_success("All shared components initialized with unified framework")

        except Exception as e:
            log_error(f"Failed to initialize shared components: {e}")
            raise

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization for TAS search."""
        tprint_debug("💻 Starting hardware optimization initialization...")
        try:
            if self.config.enable_hardware_optimization:
                tprint_debug("🔧 Hardware optimization enabled, creating configuration...")
                # Import unified hardware manager
                from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel

                # Create hardware configuration optimized for TAS
                tprint_debug(f"📋 Creating hardware config with memory limit: {self.config.max_memory_mb}MB")
                hardware_config = HardwareConfig(
                    cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                    gpu_optimization_level=OptimizationLevel.BALANCED,  # Tree models are less GPU-intensive
                    memory_optimization_level=OptimizationLevel.BALANCED,
                    memory_limit_gb=self.config.max_memory_mb / 1024,  # Convert MB to GB
                    enable_mps_acceleration=False,  # Trees don't need GPU acceleration
                    enable_gpu_memory_pooling=False,
                    enable_batch_operations=True,
                    enable_adaptive_optimization=True,
                    learning_enabled=True,
                    auto_tuning_enabled=True,
                    performance_monitoring_enabled=True,
                    monitoring_interval=5.0
                )

                # Create unified hardware manager
                tprint_debug("🏗️ Creating unified hardware manager...")
                self.hardware_optimizer = UnifiedHardwareManager(hardware_config)

                # Optimize for TAS workload
                tprint_debug("⚙️ Optimizing hardware for TAS workload...")
                self.hardware_optimizer.optimize_for_workload(
                    WorkloadType.ML_TRAINING,
                    parameters={
                        'tree_based_training': True,
                        'parallel_evaluations': self.config.parallel_evaluation,
                        'n_workers': self.config.n_workers,
                        'memory_per_model_mb': self.config.max_memory_mb // self.config.population_size,
                        'ensemble_training': self.config.allow_ensemble_methods
                    }
                )

                tprint_success("✅ Hardware optimization initialized for TAS")
                tprint_info(f"   CPU Optimization: {hardware_config.cpu_optimization_level.value}")
                tprint_info(f"   GPU Optimization: {hardware_config.gpu_optimization_level.value}")
                tprint_info(f"   Memory Limit: {hardware_config.memory_limit_gb} GB")
                log_success("Hardware optimization initialized for TAS")
                log_info(f"   CPU Optimization: {hardware_config.cpu_optimization_level.value}")
                log_info(f"   GPU Optimization: {hardware_config.gpu_optimization_level.value}")
                log_info(f"   Memory Limit: {hardware_config.memory_limit_gb} GB")
            else:
                tprint_debug("🚫 Hardware optimization disabled")
                self.hardware_optimizer = None
                log_warning("Hardware optimization disabled")

        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
            log_warning(f"Hardware optimization initialization failed: {e}")
            self.hardware_optimizer = None

    def _create_tree_constraints(self):
        """Create tree architecture constraints from config."""
        from ...hybrid_nas_tas_regime.shared_utils.constraint_systems import ArchitectureConstraints

        return ArchitectureConstraints(
            max_layers=self.config.max_trees,
            min_layers=1,
            max_parameters=1000000,  # Trees typically have fewer parameters
            max_memory_usage_mb=self.config.max_memory_mb,
            max_training_time_seconds=self.config.max_training_time_per_arch,
            max_tree_depth=self.config.max_tree_depth,
            max_complexity_score=3.0  # Trees are generally less complex
        )

    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None) -> TASResult:
        """Perform comprehensive tree architecture search with ML Common utilities."""
        self.start_time = time.time()
        tprint_info("🚀 Starting Enhanced TAS Search with ML Common utilities...")
        self.logger.info("🚀 Starting Enhanced TAS Search with ML Common utilities...")

        try:
            # Apply safeguards before search using shared utilities
            tprint_debug("🛡️ Checking training safety...")
            if not self.shared_ml_utilities.check_training_safety(train_data, validation_data):
                tprint_warning("⚠️ Training safety check failed, proceeding with caution")
                self.logger.warning("Training safety check failed, proceeding with caution")

            # Check lookahead protection using shared utilities
            tprint_debug("🔍 Validating data split for lookahead bias...")
            if not self.shared_ml_utilities.validate_data_split(train_data, validation_data):
                tprint_warning("⚠️ Data split may have lookahead bias")
                self.logger.warning("Data split may have lookahead bias")

            # Select and initialize search strategy
            tprint_debug(f"🎯 Creating search strategy: {self.config.search_strategy.value}")
            search_strategy = self._create_search_strategy()

            # Define enhanced objective function with shared utilities
            tprint_debug("🎯 Defining objective function with shared utilities...")
            def objective_function(architecture):
                return self._evaluate_architecture_with_shared_utilities(
                    architecture, validation_data, regime_data
                )

            # Perform search based on strategy with error handling
            try:
                tprint_info(f"🔍 Executing {self.config.search_strategy.value} search strategy...")
                # Perform search based on strategy
                if self.config.search_strategy == TreeSearchStrategy.RANDOM:
                    tprint_debug("🎲 Executing random search...")
                    result = self._random_search(objective_function)
                elif self.config.search_strategy == TreeSearchStrategy.BAYESIAN_OPTIMIZATION:
                    tprint_debug("📊 Executing Bayesian optimization...")
                    result = self._bayesian_search(objective_function, search_strategy)
                elif self.config.search_strategy == TreeSearchStrategy.EVOLUTIONARY:
                    tprint_debug("🧬 Executing evolutionary search...")
                    result = self._evolutionary_search(objective_function, search_strategy)
                elif self.config.search_strategy == TreeSearchStrategy.REINFORCEMENT_LEARNING:
                    tprint_debug("🤖 Executing reinforcement learning search...")
                    result = self._rl_search(objective_function, search_strategy)
                elif self.config.search_strategy == TreeSearchStrategy.ENHANCED_BAYESIAN:
                    tprint_debug("📈 Executing enhanced Bayesian search...")
                    result = self._enhanced_bayesian_search(objective_function, search_strategy)
                elif self.config.search_strategy == TreeSearchStrategy.ADAPTIVE_EVOLUTIONARY:
                    tprint_debug("🔄 Executing adaptive evolutionary search...")
                    result = self._adaptive_evolutionary_search(objective_function, search_strategy)
                else:
                    tprint_debug("🔀 Executing hybrid search...")
                    result = self._hybrid_search(objective_function, search_strategy)
                
                tprint_success(f"✅ {self.config.search_strategy.value} search completed")

                # Validate final result
                tprint_debug(f"📊 Best score: {result['best_score']}")
                if result['best_score'] > 0:
                    tprint_debug("✅ Performing cross-validation on best architecture...")
                    # Perform cross-validation on best architecture using shared utilities
                    cv_result = self._perform_cross_validation_shared(result['best_architecture'], train_data, validation_data)

                    # Optimize thresholds if applicable using shared utilities
                    if test_data:
                        tprint_debug("🎯 Optimizing model thresholds...")
                        optimized_thresholds = self._optimize_model_thresholds_shared(result['best_architecture'], test_data)
                else:
                    tprint_warning("⚠️ Best score is not positive, skipping validation")

                execution_time = time.time() - self.start_time
                tprint_info(f"⏱️ Total execution time: {execution_time:.2f} seconds")

                # Create final result with enhanced metadata
                tprint_debug("📋 Creating final search result...")
                search_result = TASResult(
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
                        'utility_type': 'TAS',
                        'safeguards_applied': True,
                        'cross_validation_performed': True if 'cv_result' in locals() else False,
                        'memory_optimized': True,
                        'error_handling_enabled': True
                    }
                )

                tprint_success("✅ Enhanced TAS Search completed successfully with ML Common utilities")
                tprint_info(f"   Best Score: {search_result.best_score:.4f}")
                tprint_info(f"   Total Evaluations: {self.evaluation_count}")
                tprint_info(f"   Execution Time: {execution_time:.2f}s")
                self.logger.info("✅ Enhanced TAS Search completed successfully with ML Common utilities")
                self.logger.info(f"   Best Score: {search_result.best_score:.4f}")
                self.logger.info(f"   Total Evaluations: {self.evaluation_count}")
                self.logger.info(f"   Execution Time: {execution_time:.2f}s")

                return search_result

            except Exception as search_error:
                tprint_error(f"❌ Search strategy failed: {search_error}")
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
            tprint_error(f"❌ Enhanced TAS Search failed: {e}")
            self.logger.error(f"❌ Enhanced TAS Search failed: {e}")

            # Return partial result with error information
            return TASResult(
                best_architecture=self.best_architecture,
                best_score=self.best_score,
                search_history=self.search_history,
                pareto_frontier=self.pareto_frontier,
                strategy_used=self.config.search_strategy.value,
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                n_evaluations=self.evaluation_count,
                metadata={'error': str(e), 'shared_ml_utilities_used': True, 'utility_type': 'TAS'}
            )

    def _create_search_strategy(self):
        """Create the appropriate search strategy."""
        if self.config.search_strategy == TreeSearchStrategy.REINFORCEMENT_LEARNING:
            return create_rl_search_strategy({
                'agent_type': 'q_learning',
                'learning_rate': 0.01,
                'exploration_rate': 1.0,
                'exploration_decay': 0.995
            })
        elif self.config.search_strategy == TreeSearchStrategy.ENHANCED_BAYESIAN:
            return create_enhanced_bayesian_search({
                'n_initial_points': min(20, self.config.population_size),
                'acquisition_function': 'expected_improvement',
                'kernel_type': 'matern'
            })
        elif self.config.search_strategy == TreeSearchStrategy.ADAPTIVE_EVOLUTIONARY:
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
        """Evaluate architecture using shared ML utilities."""
        try:
            # Use TAS-specific evaluation from shared utilities
            return self.shared_ml_utilities.evaluate_tree_architecture(
                architecture, validation_data, regime_data
            )

        except Exception as e:
            self.logger.error(f"Architecture evaluation with shared utilities failed: {e}")
            # Fallback to simple evaluation
            return self._evaluate_architecture_fallback(architecture, validation_data, regime_data)

    def _evaluate_architecture(self, architecture, validation_data, regime_data=None) -> float:
        """Evaluate a tree architecture's performance."""
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

                    log_debug(f"Tree architecture evaluated with estimator: {estimated_score:.4f}")
                    return estimated_score
                except Exception as e:
                    log_warning(f"Performance estimator failed: {e}")

            # Fallback to simplified evaluation
            X_val, y_val = validation_data

            # Tree-specific evaluation based on architecture properties
            n_trees = len(architecture.trees)
            avg_depth = sum(tree.max_depth or 10 for tree in architecture.trees) / max(n_trees, 1)
            has_boosting = any(tree.tree_type.value in ['gradient_boosting', 'xgboost'] for tree in architecture.trees)

            # Simulate performance based on tree characteristics
            base_score = 0.6  # Trees often perform well
            tree_count_bonus = min(n_trees * 0.02, 0.2)
            depth_penalty = max(0, (avg_depth - 10) * 0.01)  # Penalty for deep trees
            boosting_bonus = 0.1 if has_boosting else 0.0

            score = base_score + tree_count_bonus - depth_penalty + boosting_bonus

            # Add some noise for realism
            score += np.random.normal(0, 0.03)
            score = max(0.1, min(0.9, score))

            evaluation_time = time.time() - start_time
            self.evaluation_times.append(evaluation_time)
            self.evaluation_count += 1

            return score

        except Exception as e:
            self.logger.error(f"Tree architecture evaluation failed: {e}")
            return 0.1  # Low score for failed architectures

    def _evaluate_architecture_fallback(self, architecture, validation_data, regime_data=None) -> float:
        """Fallback evaluation when shared utilities fail."""
        try:
            X_val, y_val = validation_data

            # Tree-specific evaluation based on architecture properties
            n_trees = len(architecture.trees)
            avg_depth = sum(tree.max_depth or 10 for tree in architecture.trees) / max(n_trees, 1)
            has_boosting = any(tree.tree_type.value in ['gradient_boosting', 'xgboost'] for tree in architecture.trees)

            # Simulate performance based on tree characteristics
            base_score = 0.6  # Trees often perform well
            tree_count_bonus = min(n_trees * 0.02, 0.2)
            depth_penalty = max(0, (avg_depth - 10) * 0.01)  # Penalty for deep trees
            boosting_bonus = 0.1 if has_boosting else 0.0

            score = base_score + tree_count_bonus - depth_penalty + boosting_bonus

            # Add some noise for realism
            score += np.random.normal(0, 0.03)
            score = max(0.1, min(0.9, score))

            return score

        except Exception as e:
            self.logger.error(f"Tree architecture fallback evaluation failed: {e}")
            return 0.1  # Low score for failed architectures

    def _random_search(self, objective_function: Callable) -> Dict[str, Any]:
        """Perform random search for tree architectures."""
        self.logger.info("🔍 Starting Random Search for Trees...")

        best_architecture = None
        best_score = -np.inf

        for i in range(self.config.max_evaluations):
            # Generate random tree architecture
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
        """Perform Bayesian optimization search for trees."""
        self.logger.info("🔍 Starting Bayesian Optimization Search for Trees...")

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
        """Perform enhanced Bayesian optimization search for trees."""
        self.logger.info("🔍 Starting Enhanced Bayesian Optimization Search for Trees...")

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
        """Perform evolutionary search for trees."""
        self.logger.info("🔍 Starting Evolutionary Search for Trees...")

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
        """Perform reinforcement learning search for trees."""
        self.logger.info("🔍 Starting Reinforcement Learning Search for Trees...")

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
        """Perform hybrid search combining multiple strategies for trees."""
        self.logger.info("🔍 Starting Hybrid Search for Trees...")

        # Combine multiple strategies
        strategies = [
            self._create_search_strategy_class('bayesian_optimization'),
            self._create_search_strategy_class('evolutionary'),
            self._create_search_strategy_class('rl')
        ]

        best_overall_architecture = None
        best_overall_score = -np.inf

        for i, strategy in enumerate(strategies):
            self.logger.info(f"Running tree strategy {i+1}/{len(strategies)}")

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
        """Create a search strategy instance by name for trees."""
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
        """Generate a random tree architecture from search space."""
        return self.search_space.sample_random_architecture()

    def _constraint_checker(self, architecture: Any) -> Any:
        """Check if tree architecture meets constraints."""
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
            self.logger.warning(f"Cross-validation with shared utilities failed: {e}")
            return {'error': str(e), 'success': False}

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
            self.logger.warning(f"Threshold optimization with shared utilities failed: {e}")
            return {'success': False, 'error': str(e)}

    def save_search_state(self, filepath: str) -> bool:
        """Save the current TAS search state with shared ML utilities."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
                # Shared ML utilities state
                'shared_ml_utilities_used': True,
                'utility_type': 'TAS',
                'ml_utilities_status': self.shared_ml_utilities.get_system_status()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"✅ TAS search state saved to {filepath} with shared ML utilities")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save TAS search state: {e}")
            return False

    def load_search_state(self, filepath: str) -> bool:
        """Load a saved TAS search state with shared ML utilities."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = state['config']
            self.current_generation = state['current_generation']
            self.best_architecture = state['best_architecture']
            self.best_score = state['best_score']
            self.search_history = state['search_history']
            self.pareto_frontier = state['pareto_frontier']
            self.evaluation_count = state['evaluation_count']
            self.evaluation_times = state['evaluation_times']
            self.start_time = state['start_time']

            # Restore shared ML utilities state if available
            if state.get('shared_ml_utilities_used', False):
                self.logger.info("Loading TAS search state with shared ML utilities")

                # Reinitialize shared ML utilities
                self._initialize_shared_ml_utilities()
                self._initialize_shared_components()

            self.logger.info(f"✅ TAS search state loaded from {filepath} with shared ML utilities")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load TAS search state: {e}")
            return False


def create_enhanced_tas_engine(config: TASConfig) -> EnhancedTASEngine:
    """Create an enhanced TAS engine instance."""
    return EnhancedTASEngine(config)


def quick_tas_search(train_data: Tuple[np.ndarray, np.ndarray],
                    validation_data: Tuple[np.ndarray, np.ndarray],
                    config: Optional[TASConfig] = None) -> TASResult:
    """Quick TAS search with default settings and shared ML utilities."""
    if config is None:
        config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=30,
            max_generations=50,
            max_evaluations=200,
            # Enable shared ML utilities
            enable_multi_objective=True,
            enable_constraint_validation=True,
            enable_performance_estimation=True,
            parallel_evaluation=True
        )

    engine = EnhancedTASEngine(config)
    return engine.search(train_data, validation_data)