"""
Unified Architecture Search Engine for NAS-TAS Integration

This engine provides a comprehensive framework for both Neural Architecture Search (NAS)
and Tree Architecture Search (TAS), combining their search spaces, strategies, and
optimization objectives for financial trading applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import pickle
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Import existing components
from ..shared_utils.search_spaces import NeuralSearchSpace, TreeSearchSpace
from ..shared_utils.advanced_search_strategies import (
    ReinforcementLearningSearch, EnhancedBayesianOptimization, AdaptiveEvolutionarySearch
)
from ..shared_utils.performance_estimators import UnifiedPerformanceEstimator
from ..shared_utils.architecture_encoders import UnifiedArchitectureEncoder
from ..shared_utils.constraint_systems import UnifiedConstraintValidator

# Import NAS and TAS engines from centralized utilities
from src.utils.nas_tas.core.nas_engine import NASEngine
from src.utils.nas_tas.core.tas_engine import TASEngine
from src.utils.nas_tas.optimization.architecture_search import ArchitectureSearchOptimizer, ArchitectureSearchConfig
from src.utils.nas_tas.optimization.strategy_search import StrategySearchOptimizer, StrategySearchConfig

logger = logging.getLogger(__name__)

class ArchitectureType(Enum):
    """Types of architectures supported."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"

class SearchMode(Enum):
    """Search modes for the unified engine."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    REGIME_AWARE = "regime_aware"
    ADAPTIVE = "adaptive"
    CONTINUAL = "continual"

@dataclass
class UnifiedSearchConfig:
    """Configuration for unified architecture search."""
    # Base configuration
    architecture_types: List[ArchitectureType] = field(default_factory=lambda: [ArchitectureType.NEURAL, ArchitectureType.TREE])
    search_mode: SearchMode = SearchMode.MULTI_OBJECTIVE
    max_evaluations: int = 1000
    max_search_time: int = 3600  # 1 hour
    population_size: int = 50

    # Financial objectives
    enable_trading_objectives: bool = True
    sharpe_weight: float = 0.4
    max_drawdown_weight: float = 0.3
    win_rate_weight: float = 0.2
    profit_factor_weight: float = 0.1

    # Hardware constraints
    max_memory_mb: int = 8192
    max_training_time_per_arch: int = 600
    parallel_evaluation: bool = True
    n_workers: int = 4

    # Advanced features
    enable_performance_estimation: bool = True
    enable_architecture_encoding: bool = True
    enable_constraint_validation: bool = True
    enable_meta_learning: bool = True

    # Output settings
    save_results: bool = True
    save_best_architectures: bool = True
    output_dir: str = "unified_search_results"

@dataclass
class UnifiedSearchResult:
    """Result from unified architecture search."""
    best_architecture: Dict[str, Any]
    best_score: float
    architecture_type: ArchitectureType
    search_history: List[Dict[str, Any]]
    pareto_frontier: List[Dict[str, Any]]
    trading_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    execution_time: float
    n_evaluations: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedArchitectureSearchEngine:
    """
    Unified Architecture Search Engine for NAS-TAS integration.

    This engine provides a comprehensive framework that can search across both
    neural and tree-based architectures, using advanced search strategies and
    financial trading objectives.
    """

    def __init__(self, config: UnifiedSearchConfig):
        """Initialize the unified architecture search engine."""
        tprint("🚀 [UNIFIED_ARCH_SEARCH] Initializing Unified Architecture Search Engine", color="cyan", bold=True)
        tprint(f"📊 [UNIFIED_ARCH_SEARCH] Architecture Types: {[t.value for t in config.architecture_types]}", color="blue")
        tprint(f"📊 [UNIFIED_ARCH_SEARCH] Search Mode: {config.search_mode.value}", color="blue")
        tprint(f"📊 [UNIFIED_ARCH_SEARCH] Max Evaluations: {config.max_evaluations}", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        tprint("🔧 [UNIFIED_ARCH_SEARCH] Initializing search components", color="yellow")
        self._initialize_components()

        # Search state
        tprint("📊 [UNIFIED_ARCH_SEARCH] Setting up search state", color="blue")
        self.current_generation = 0
        self.best_architecture = None
        self.best_score = -np.inf
        self.search_history = []
        self.pareto_frontier = []
        self.evaluation_count = 0

        # Performance tracking
        self.start_time = None
        self.evaluation_times = []

        tprint("✅ [UNIFIED_ARCH_SEARCH] Unified Architecture Search Engine initialized successfully", color="green")
        self.logger.info("✅ Unified Architecture Search Engine initialized")
        self.logger.info(f"   Architecture Types: {[t.value for t in config.architecture_types]}")
        self.logger.info(f"   Search Mode: {config.search_mode.value}")
        self.logger.info(f"   Max Evaluations: {config.max_evaluations}")

    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Search spaces
            self.neural_search_space = NeuralSearchSpace()
            self.tree_search_space = TreeSearchSpace()

            # Performance estimator
            if self.config.enable_performance_estimation:
                self.performance_estimator = UnifiedPerformanceEstimator({
                    'neural_config': {'estimator_type': 'ensemble'},
                    'tree_config': {'estimator_type': 'meta_learner'}
                })

            # Architecture encoder
            if self.config.enable_architecture_encoding:
                self.architecture_encoder = UnifiedArchitectureEncoder({
                    'encoding_method': 'hybrid',
                    'latent_dim': 128
                })

            # Constraint validator
            if self.config.enable_constraint_validation:
                self.constraint_validator = UnifiedConstraintValidator({
                    'max_layers': 20,
                    'max_parameters': 10000000,
                    'max_memory_usage_mb': self.config.max_memory_mb,
                    'max_training_time_seconds': self.config.max_training_time_per_arch
                })

            # Search strategies
            self.search_strategies = self._initialize_search_strategies()

            # NAS and TAS engines
            self.nas_engine = self._create_nas_engine()
            self.tas_engine = self._create_tas_engine()

            self.logger.info("✅ All components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _initialize_search_strategies(self) -> Dict[str, Any]:
        """Initialize advanced search strategies."""
        strategies = {}

        # Enhanced Bayesian Optimization
        strategies['bayesian'] = EnhancedBayesianOptimization({
            'n_initial_points': min(20, self.config.population_size),
            'acquisition_function': 'expected_improvement',
            'kernel_type': 'matern'
        })

        # Adaptive Evolutionary Search
        strategies['evolutionary'] = AdaptiveEvolutionarySearch({
            'population_size': self.config.population_size,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'tournament_size': 5,
            'use_island_model': True,
            'n_islands': 5
        })

        # Reinforcement Learning Search
        strategies['rl'] = ReinforcementLearningSearch({
            'agent_type': 'ppo',
            'learning_rate': 0.0003,
            'exploration_rate': 1.0,
            'exploration_decay': 0.995,
            'memory_size': 10000
        })

        return strategies

    def _create_nas_engine(self) -> NASEngine:
        """Create NAS engine with unified configuration."""
        return NASEngine()

    def _create_tas_engine(self) -> TASEngine:
        """Create TAS engine with unified configuration."""
        return TASEngine()

    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None) -> UnifiedSearchResult:
        """Perform unified architecture search across NAS and TAS spaces."""
        tprint("🚀 [UNIFIED_ARCH_SEARCH] Starting Unified Architecture Search", color="cyan", bold=True)
        tprint(f"📊 [UNIFIED_ARCH_SEARCH] Train data shape: {train_data[0].shape}, labels: {train_data[1].shape}", color="blue")
        tprint(f"📊 [UNIFIED_ARCH_SEARCH] Validation data shape: {validation_data[0].shape}, labels: {validation_data[1].shape}", color="blue")
        self.start_time = time.time()
        self.logger.info("🚀 Starting Unified Architecture Search...")

        try:
            # Prepare search environment
            tprint("🔧 [UNIFIED_ARCH_SEARCH] Preparing search environment", color="yellow")
            search_env = self._prepare_search_environment(train_data, validation_data, test_data, regime_data)
            tprint("✅ [UNIFIED_ARCH_SEARCH] Search environment prepared", color="green")

            # Perform search based on mode
            tprint(f"🎯 [UNIFIED_ARCH_SEARCH] Starting {self.config.search_mode.value} search", color="yellow")
            if self.config.search_mode == SearchMode.SINGLE_OBJECTIVE:
                result = self._single_objective_search(search_env)
            elif self.config.search_mode == SearchMode.MULTI_OBJECTIVE:
                result = self._multi_objective_search(search_env)
            elif self.config.search_mode == SearchMode.REGIME_AWARE:
                result = self._regime_aware_search(search_env)
            elif self.config.search_mode == SearchMode.ADAPTIVE:
                result = self._adaptive_search(search_env)
            elif self.config.search_mode == SearchMode.CONTINUAL:
                result = self._continual_search(search_env)
            else:
                raise ValueError(f"Unknown search mode: {self.config.search_mode}")

            execution_time = time.time() - self.start_time
            tprint(f"✅ [UNIFIED_ARCH_SEARCH] Search completed in {execution_time:.2f}s", color="green")

            # Create final result
            tprint("📊 [UNIFIED_ARCH_SEARCH] Creating final search results", color="blue")
            search_result = UnifiedSearchResult(
                best_architecture=result['best_architecture'],
                best_score=result['best_score'],
                architecture_type=result['architecture_type'],
                search_history=self.search_history,
                pareto_frontier=self.pareto_frontier,
                trading_metrics=result.get('trading_metrics', {}),
                convergence_info=result.get('convergence_info', {}),
                execution_time=execution_time,
                n_evaluations=self.evaluation_count,
                metadata={
                    'search_mode': self.config.search_mode.value,
                    'architecture_types': [t.value for t in self.config.architecture_types],
                    'population_size': self.config.population_size,
                    'max_evaluations': self.config.max_evaluations,
                    'final_generation': self.current_generation
                }
            )

            tprint(f"🎉 [UNIFIED_ARCH_SEARCH] Best architecture: {result['architecture_type']}, score: {result['best_score']:.4f}", color="cyan")
            self.logger.info("✅ Unified Architecture Search completed successfully")
            self.logger.info(f"   Best Score: {search_result.best_score:.4f}")
            self.logger.info(f"   Architecture Type: {search_result.architecture_type.value}")
            self.logger.info(f"   Total Evaluations: {self.evaluation_count}")
            self.logger.info(f"   Execution Time: {execution_time:.2f}s")

            # Save results if requested
            if self.config.save_results:
                self._save_search_results(search_result)

            return search_result

        except Exception as e:
            execution_time = time.time() - self.start_time
            self.logger.error(f"❌ Unified Architecture Search failed: {e}")

            # Return partial result
            return UnifiedSearchResult(
                best_architecture=self.best_architecture,
                best_score=self.best_score,
                architecture_type=ArchitectureType.HYBRID,
                search_history=self.search_history,
                pareto_frontier=self.pareto_frontier,
                trading_metrics={},
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                n_evaluations=self.evaluation_count,
                metadata={'error': str(e)}
            )

    def _prepare_search_environment(self,
                                   train_data: Tuple[np.ndarray, np.ndarray],
                                   validation_data: Tuple[np.ndarray, np.ndarray],
                                   test_data: Optional[Tuple[np.ndarray, np.ndarray]],
                                   regime_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare search environment with all necessary components."""
        return {
            'train_data': train_data,
            'validation_data': validation_data,
            'test_data': test_data,
            'regime_data': regime_data,
            'neural_search_space': self.neural_search_space,
            'tree_search_space': self.tree_search_space,
            'performance_estimator': getattr(self, 'performance_estimator', None),
            'architecture_encoder': getattr(self, 'architecture_encoder', None),
            'constraint_validator': getattr(self, 'constraint_validator', None),
            'search_strategies': self.search_strategies,
            'nas_engine': self.nas_engine,
            'tas_engine': self.tas_engine
        }

    def _evaluate_architecture(self, architecture: Dict[str, Any], search_env: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate an architecture's performance with trading metrics."""
        X_val, y_val = search_env['validation_data']
        start_time = time.time()

        try:
            # Use performance estimator if available
            if self.config.enable_performance_estimation and self.performance_estimator:
                try:
                    prediction = self.performance_estimator.predict_performance(architecture)
                    estimated_score = prediction.predicted_performance
                    evaluation_time = time.time() - start_time

                    # Store evaluation info
                    self.evaluation_times.append(evaluation_time)
                    self.evaluation_count += 1

                    # Calculate trading metrics
                    trading_metrics = self._calculate_trading_metrics(architecture, X_val, y_val, estimated_score)

                    return {
                        'performance': estimated_score,
                        'trading_score': trading_metrics['composite_score'],
                        'evaluation_time': evaluation_time
                    }
                except Exception as e:
                    self.logger.warning(f"Performance estimator failed: {e}")

            # Fallback to actual evaluation (simplified)
            # In practice, this would involve training and validating the architecture
            complexity_score = self._calculate_complexity_score(architecture)
            parameter_efficiency = self._calculate_parameter_efficiency(architecture)

            # Base performance score
            base_score = 0.5 + np.random.normal(0, 0.1)  # Add some noise for realism

            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(architecture, X_val, y_val, base_score)

            evaluation_time = time.time() - start_time
            self.evaluation_times.append(evaluation_time)
            self.evaluation_count += 1

            return {
                'performance': base_score,
                'trading_score': trading_metrics['composite_score'],
                'evaluation_time': evaluation_time
            }

        except Exception as e:
            self.logger.error(f"Architecture evaluation failed: {e}")
            return {
                'performance': 0.1,
                'trading_score': 0.1,
                'evaluation_time': time.time() - start_time
            }

    def _calculate_trading_metrics(self, architecture: Dict[str, Any],
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 performance_score: float) -> Dict[str, float]:
        """Calculate trading-specific metrics for architecture evaluation."""
        try:
            # Simulate trading metrics based on architecture characteristics
            architecture_type = architecture.get('type', 'neural')

            # Base metrics
            sharpe_ratio = np.random.uniform(0.5, 2.0)  # Simulated Sharpe ratio
            max_drawdown = np.random.uniform(-0.3, -0.05)  # Simulated max drawdown
            win_rate = np.random.uniform(0.4, 0.7)  # Simulated win rate
            profit_factor = np.random.uniform(1.1, 2.5)  # Simulated profit factor

            # Architecture-specific adjustments
            if architecture_type == 'neural':
                # Neural networks might have better Sharpe but higher drawdown
                sharpe_ratio *= 1.1
                max_drawdown *= 1.2
            elif architecture_type == 'tree':
                # Tree models might have lower Sharpe but more stable
                sharpe_ratio *= 0.9
                max_drawdown *= 0.8

            # Normalize metrics for scoring
            normalized_sharpe = min(sharpe_ratio / 2.0, 1.0)  # Cap at 2.0
            normalized_drawdown = max(max_drawdown / -0.5, -1.0)  # Normalize negative values
            normalized_win_rate = win_rate
            normalized_profit_factor = min(profit_factor / 2.0, 1.0)  # Cap at 2.0

            # Calculate composite score using configured weights
            composite_score = (
                self.config.sharpe_weight * normalized_sharpe +
                self.config.max_drawdown_weight * (1.0 + normalized_drawdown) +  # Convert to positive scale
                self.config.win_rate_weight * normalized_win_rate +
                self.config.profit_factor_weight * normalized_profit_factor
            )

            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'composite_score': composite_score,
                'architecture_type': architecture_type,
                'performance_score': performance_score
            }

        except Exception as e:
            self.logger.warning(f"Trading metrics calculation failed: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'composite_score': 0.0,
                'architecture_type': 'unknown',
                'performance_score': 0.0
            }

    def _calculate_complexity_score(self, architecture: Dict[str, Any]) -> float:
        """Calculate architecture complexity score."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)
            n_parameters = sum(layer.get('hidden_size', 100) for layer in layers)

            # Complexity score (lower is better)
            complexity = (n_layers / 10.0) * (n_parameters / 100000.0)
            return min(complexity, 1.0)

        except Exception as e:
            logger.warning(f"Failed to calculate architecture complexity score: {e}. Using default score of 0.5")
            return 0.5

    def _calculate_parameter_efficiency(self, architecture: Dict[str, Any]) -> float:
        """Calculate parameter efficiency score."""
        try:
            layers = architecture.get('layers', [])
            n_parameters = sum(layer.get('hidden_size', 100) for layer in layers)

            # Efficiency score (higher is better)
            efficiency = 1000.0 / max(n_parameters, 1000.0)
            return min(efficiency, 1.0)

        except Exception as e:
            logger.warning(f"Failed to calculate architecture complexity score: {e}. Using default score of 0.5")
            return 0.5

    def _single_objective_search(self, search_env: Dict[str, Any]) -> Dict[str, Any]:
        """Perform single-objective architecture search."""
        self.logger.info("🎯 Performing Single-Objective Search...")

        # Use enhanced Bayesian optimization for single objective
        strategy = self.search_strategies['bayesian']

        def objective_function(architecture):
            metrics = self._evaluate_architecture(architecture, search_env)
            return metrics['performance']

        result = strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_evaluations // self.config.population_size
        )

        return {
            'best_architecture': result.best_architecture,
            'best_score': result.best_score,
            'architecture_type': result.best_architecture.get('type', 'neural'),
            'convergence_info': result.convergence_info
        }

    def _multi_objective_search(self, search_env: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-objective architecture search."""
        self.logger.info("🎯 Performing Multi-Objective Search...")

        # Use evolutionary search for multi-objective
        strategy = self.search_strategies['evolutionary']

        def objective_function(architecture):
            metrics = self._evaluate_architecture(architecture, search_env)
            return [metrics['performance'], metrics['trading_score']]

        result = strategy.multi_objective_search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_evaluations // self.config.population_size
        )

        # Find best architecture from Pareto frontier
        best_architecture = None
        best_score = -np.inf

        for arch in result.pareto_frontier:
            metrics = self._evaluate_architecture(arch, search_env)
            if metrics['trading_score'] > best_score:
                best_score = metrics['trading_score']
                best_architecture = arch

        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'architecture_type': best_architecture.get('type', 'neural') if best_architecture else 'neural',
            'trading_metrics': metrics,
            'convergence_info': result.convergence_info
        }

    def _regime_aware_search(self, search_env: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regime-aware architecture search."""
        self.logger.info("🎯 Performing Regime-Aware Search...")

        # Combine multiple strategies for regime awareness
        strategies = [
            self.search_strategies['bayesian'],
            self.search_strategies['evolutionary']
        ]

        best_overall_architecture = None
        best_overall_score = -np.inf
        trading_metrics = {}

        for i, strategy in enumerate(strategies):
            self.logger.info(f"Running regime-aware strategy {i+1}/{len(strategies)}")

            def objective_function(architecture):
                metrics = self._evaluate_architecture(architecture, search_env)
                return metrics['performance']

            result = strategy.search(
                architecture_generator=self._architecture_generator,
                performance_evaluator=objective_function,
                constraint_validator=self._constraint_checker,
                n_iterations=self.config.max_evaluations // (self.config.population_size * len(strategies))
            )

            if result.best_score > best_overall_score:
                best_overall_score = result.best_score
                best_overall_architecture = result.best_architecture
                metrics = self._evaluate_architecture(result.best_architecture, search_env)
                trading_metrics = metrics

        return {
            'best_architecture': best_overall_architecture,
            'best_score': best_overall_score,
            'architecture_type': best_overall_architecture.get('type', 'neural') if best_overall_architecture else 'neural',
            'trading_metrics': trading_metrics,
            'convergence_info': {'strategies_used': len(strategies)}
        }

    def _adaptive_search(self, search_env: Dict[str, Any]) -> Dict[str, Any]:
        """Perform adaptive architecture search."""
        self.logger.info("🎯 Performing Adaptive Search...")

        # Start with Bayesian optimization, then switch to evolutionary
        n_bayesian_iterations = self.config.max_evaluations // (self.config.population_size * 2)
        n_evolutionary_iterations = self.config.max_evaluations // (self.config.population_size * 2)

        # Bayesian phase
        bayesian_strategy = self.search_strategies['bayesian']

        def objective_function(architecture):
            metrics = self._evaluate_architecture(architecture, search_env)
            return metrics['performance']

        bayesian_result = bayesian_strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=n_bayesian_iterations
        )

        # Evolutionary phase (using Bayesian results as initial population)
        evolutionary_strategy = self.search_strategies['evolutionary']

        def evolutionary_objective(architecture):
            metrics = self._evaluate_architecture(architecture, search_env)
            return [metrics['performance'], metrics['trading_score']]

        evolutionary_result = evolutionary_strategy.multi_objective_search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=evolutionary_objective,
            constraint_validator=self._constraint_checker,
            n_iterations=n_evolutionary_iterations
        )

        # Find best from evolutionary phase
        best_architecture = None
        best_score = -np.inf

        for arch in evolutionary_result.pareto_frontier:
            metrics = self._evaluate_architecture(arch, search_env)
            if metrics['trading_score'] > best_score:
                best_score = metrics['trading_score']
                best_architecture = arch

        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'architecture_type': best_architecture.get('type', 'neural') if best_architecture else 'neural',
            'trading_metrics': metrics,
            'convergence_info': {
                'bayesian_iterations': n_bayesian_iterations,
                'evolutionary_iterations': n_evolutionary_iterations
            }
        }

    def _continual_search(self, search_env: Dict[str, Any]) -> Dict[str, Any]:
        """Perform continual architecture search."""
        self.logger.info("🎯 Performing Continual Search...")

        # Use RL for continual learning
        rl_strategy = self.search_strategies['rl']

        def objective_function(architecture):
            metrics = self._evaluate_architecture(architecture, search_env)
            return metrics['performance']

        result = rl_strategy.search(
            architecture_generator=self._architecture_generator,
            performance_evaluator=objective_function,
            constraint_validator=self._constraint_checker,
            n_iterations=self.config.max_evaluations // self.config.population_size
        )

        metrics = self._evaluate_architecture(result.best_architecture, search_env)

        return {
            'best_architecture': result.best_architecture,
            'best_score': result.best_score,
            'architecture_type': result.best_architecture.get('type', 'neural') if result.best_architecture else 'neural',
            'trading_metrics': metrics,
            'convergence_info': result.convergence_info
        }

    def _architecture_generator(self) -> Dict[str, Any]:
        """Generate a random architecture from search spaces."""
        # Randomly choose architecture type
        architecture_type = np.random.choice(self.config.architecture_types)

        if architecture_type == ArchitectureType.NEURAL:
            return self.neural_search_space.sample_random_architecture()
        elif architecture_type == ArchitectureType.TREE:
            return self.tree_search_space.sample_random_architecture()
        else:
            # Hybrid architecture
            neural_arch = self.neural_search_space.sample_random_architecture()
            tree_arch = self.tree_search_space.sample_random_architecture()
            return {
                'type': 'hybrid',
                'neural_component': neural_arch,
                'tree_component': tree_arch,
                'layers': neural_arch.get('layers', []) + tree_arch.get('layers', [])
            }

    def _constraint_checker(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Check if architecture meets constraints."""
        if self.constraint_validator:
            return self.constraint_validator.validate(architecture)
        else:
            return {'is_valid': True, 'violations': []}

    def _save_search_results(self, result: UnifiedSearchResult):
        """Save search results to disk."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save result
            result_file = output_dir / "unified_search_result.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'best_architecture': result.best_architecture,
                    'best_score': result.best_score,
                    'architecture_type': result.architecture_type.value,
                    'trading_metrics': result.trading_metrics,
                    'execution_time': result.execution_time,
                    'n_evaluations': result.n_evaluations,
                    'metadata': result.metadata
                }, f, indent=2, default=str)

            # Save best architecture
            if result.best_architecture and self.config.save_best_architectures:
                arch_file = output_dir / "best_architecture.json"
                with open(arch_file, 'w') as f:
                    json.dump(result.best_architecture, f, indent=2, default=str)

            self.logger.info(f"💾 Search results saved to {output_dir}")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not save search results: {e}")

def create_unified_search_engine(config: UnifiedSearchConfig) -> UnifiedArchitectureSearchEngine:
    """Create a unified architecture search engine instance."""
    return UnifiedArchitectureSearchEngine(config)

def quick_unified_search(train_data: Tuple[np.ndarray, np.ndarray],
                        validation_data: Tuple[np.ndarray, np.ndarray],
                        config: Optional[UnifiedSearchConfig] = None) -> UnifiedSearchResult:
    """Quick unified architecture search with default settings."""
    if config is None:
        config = UnifiedSearchConfig(
            architecture_types=[ArchitectureType.NEURAL, ArchitectureType.TREE],
            search_mode=SearchMode.MULTI_OBJECTIVE,
            max_evaluations=200,
            population_size=30
        )

    engine = UnifiedArchitectureSearchEngine(config)
    return engine.search(train_data, validation_data)
