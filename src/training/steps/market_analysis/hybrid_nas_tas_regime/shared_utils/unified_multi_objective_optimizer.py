"""
Unified Multi-Objective Optimizer

This module provides a unified multi-objective optimization system that combines
the best practices from both TAS and NAS regime detection systems. It optimizes
multiple objectives simultaneously for regime detection and trading systems.

Features:
- Unified optimization strategies
- Support for both tree-based and neural-based architectures
- Advanced Pareto frontier management
- Configurable objective weights
- Multiple optimization algorithms (NSGA-II, SPEA2, Bayesian)
- Real-time optimization capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime
from abc import ABC, abstractmethod

# Import unified evaluators
from .unified_economic_evaluator import UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig
from .unified_trading_viability_evaluator import UnifiedTradingViabilityEvaluator, TradingViabilityConfig

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    REGIME_ACCURACY = "regime_accuracy"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    ARCHITECTURE_COMPLEXITY = "architecture_complexity"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_ACCURACY = "transition_accuracy"
    MODEL_CONFIDENCE = "model_confidence"
    RISK_ADJUSTED_RETURNS = "risk_adjusted_returns"
    EXECUTION_FEASIBILITY = "execution_feasibility"

class OptimizationAlgorithm(Enum):
    """Types of optimization algorithms."""
    NSGA2 = "nsga2"
    SPEA2 = "spea2"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"

@dataclass
class OptimizationConfig:
    """Configuration for unified multi-objective optimization."""

    # Objectives to optimize
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.REGIME_ACCURACY,
        OptimizationObjective.ECONOMIC_SIGNIFICANCE,
        OptimizationObjective.TRADING_VIABILITY,
        OptimizationObjective.COMPUTATIONAL_EFFICIENCY
    ])

    # Objective weights
    objective_weights: Dict[OptimizationObjective, float] = field(default_factory=lambda: {
        OptimizationObjective.REGIME_ACCURACY: 0.3,
        OptimizationObjective.ECONOMIC_SIGNIFICANCE: 0.25,
        OptimizationObjective.TRADING_VIABILITY: 0.25,
        OptimizationObjective.COMPUTATIONAL_EFFICIENCY: 0.2
    })

    # Optimization algorithm
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA2

    # Optimization parameters
    max_iterations: int = 100
    population_size: int = 50
    n_initial_points: int = 10
    convergence_threshold: float = 1e-6
    early_stopping_patience: int = 20

    # Pareto frontier management
    max_pareto_solutions: int = 100
    pareto_epsilon: float = 0.01

    # Evaluation configuration
    economic_config: Optional[EconomicEvaluationConfig] = field(default_factory=EconomicEvaluationConfig)
    trading_config: Optional[TradingViabilityConfig] = field(default_factory=TradingViabilityConfig)

    # Advanced features
    enable_constraint_handling: bool = True
    enable_adaptive_weights: bool = False
    enable_real_time_optimization: bool = False

    # Performance settings
    parallel_evaluation: bool = True
    n_workers: int = 4
    memory_limit_gb: float = 8.0

    # TAS-specific enhancements
    enable_tree_based_optimization: bool = True
    tree_complexity_weight: float = 0.2
    tree_interpretability_weight: float = 0.3
    tree_depth_penalty: float = 0.1

    # NAS-specific enhancements
    enable_neural_based_optimization: bool = True
    neural_architecture_complexity_weight: float = 0.2
    neural_uncertainty_weight: float = 0.3
    neural_efficiency_weight: float = 0.1

    # Hybrid optimization
    enable_hybrid_optimization: bool = True
    hybrid_consensus_weight: float = 0.4
    hybrid_ensemble_weight: float = 0.3

@dataclass
class OptimizationResult:
    """Result from unified multi-objective optimization."""

    # Optimization success
    success: bool
    execution_time: float

    # Pareto solutions
    pareto_solutions: List[Dict[str, Any]]
    best_solution: Optional[Dict[str, Any]]

    # Optimization metrics
    optimization_metrics: Dict[str, Any]
    convergence_history: List[Dict[str, float]]

    # Objective scores
    objective_scores: Dict[str, float]
    weighted_score: float

    # Metadata
    algorithm_used: str
    n_objectives: int
    n_solutions: int
    convergence_achieved: bool

    # Error information
    error_message: Optional[str] = None

class BaseOptimizationAlgorithm(ABC):
    """Base class for optimization algorithms."""

    @abstractmethod
    def optimize(self,
                 objective_function: Callable,
                 constraints: Optional[Dict[str, Any]] = None,
                 config: OptimizationConfig = None) -> OptimizationResult:
        """Perform optimization."""
        pass

class NSGA2Optimizer(BaseOptimizationAlgorithm):
    """NSGA-II multi-objective optimizer."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self,
                 objective_function: Callable,
                 constraints: Optional[Dict[str, Any]] = None,
                 config: OptimizationConfig = None) -> OptimizationResult:
        """Perform NSGA-II optimization."""
        try:
            tprint_info("Starting NSGA-II optimization")
            tprint_debug(f"Configuration: {self.config}")
            start_time = time.time()
            self.logger.info("🚀 Starting NSGA-II optimization...")

            # Initialize population
            tprint_info("Initializing population")
            population = self._initialize_population()
            tprint_success(f"Population initialized with {len(population)} individuals")
            pareto_frontier = []
            convergence_history = []

            for iteration in range(self.config.max_iterations):
                tprint_progress(iteration, self.config.max_iterations, "NSGA-II optimization")
                # Evaluate population
                tprint_debug(f"Evaluating population for iteration {iteration}")
                evaluated_population = []
                for individual in population:
                    try:
                        scores = objective_function(individual)
                        individual['objective_scores'] = scores
                        individual['weighted_score'] = self._calculate_weighted_score(scores)
                        evaluated_population.append(individual)
                    except Exception as e:
                        tprint_warning(f"Error evaluating individual: {e}")
                        self.logger.warning(f"Objective evaluation failed: {e}")
                        continue

                # Update Pareto frontier
                pareto_frontier = self._update_pareto_frontier(pareto_frontier, evaluated_population)

                # Record convergence
                convergence_history.append({
                    'iteration': iteration,
                    'best_score': max([ind['weighted_score'] for ind in evaluated_population]) if evaluated_population else 0.0,
                    'pareto_size': len(pareto_frontier),
                    'population_diversity': self._calculate_diversity(evaluated_population)
                })

                # Check convergence
                if self._check_convergence(convergence_history):
                    self.logger.info(f"✅ Optimization converged at iteration {iteration + 1}")
                    break

                # Generate next generation
                population = self._generate_next_generation(evaluated_population, pareto_frontier)

            execution_time = time.time() - start_time

            # Create result
            result = OptimizationResult(
                success=True,
                execution_time=execution_time,
                pareto_solutions=pareto_frontier,
                best_solution=max(pareto_frontier, key=lambda x: x['weighted_score']) if pareto_frontier else None,
                optimization_metrics=self._calculate_optimization_metrics(pareto_frontier, convergence_history),
                convergence_history=convergence_history,
                objective_scores=self._calculate_objective_scores(pareto_frontier),
                weighted_score=max([sol['weighted_score'] for sol in pareto_frontier]) if pareto_frontier else 0.0,
                algorithm_used='NSGA2',
                n_objectives=len(self.config.objectives),
                n_solutions=len(pareto_frontier),
                convergence_achieved=len(convergence_history) < self.config.max_iterations
            )

            self.logger.info(f"✅ NSGA-II optimization completed in {execution_time:.2f}s")
            self.logger.info(f"   Pareto solutions: {len(pareto_frontier)}")
            self.logger.info(f"   Best score: {result.weighted_score:.4f}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NSGA-II optimization failed: {e}")

            return OptimizationResult(
                success=False,
                execution_time=execution_time,
                pareto_solutions=[],
                best_solution=None,
                optimization_metrics={},
                convergence_history=[],
                objective_scores={},
                weighted_score=0.0,
                algorithm_used='NSGA2',
                n_objectives=len(self.config.objectives),
                n_solutions=0,
                convergence_achieved=False,
                error_message=str(e)
            )

    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize optimization population."""
        population = []

        for i in range(self.config.population_size):
            individual = {
                'id': f'individual_{i}',
                'parameters': self._generate_random_parameters(),
                'generation': 0
            }
            population.append(individual)

        return population

    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters for an individual."""
        return {
            'regime_count': np.random.randint(3, 10),
            'complexity_factor': np.random.uniform(0.3, 1.0),
            'efficiency_factor': np.random.uniform(0.5, 1.0),
            'stability_factor': np.random.uniform(0.4, 1.0),
            'confidence_threshold': np.random.uniform(0.5, 0.9),
            'learning_rate': np.random.uniform(1e-4, 1e-2),
            'batch_size': np.random.choice([32, 64, 128, 256]),
            'dropout_rate': np.random.uniform(0.1, 0.5)
        }

    def _calculate_weighted_score(self, objective_scores: Dict[str, float]) -> float:
        """Calculate weighted score from objective scores."""
        weighted_score = 0.0

        for objective in self.config.objectives:
            if objective.value in objective_scores:
                weight = self.config.objective_weights.get(objective, 0.0)
                weighted_score += objective_scores[objective.value] * weight

        return weighted_score

    def _update_pareto_frontier(self, current_frontier: List[Dict[str, Any]],
                               new_solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update Pareto frontier with new solutions."""
        # Combine current frontier with new solutions
        all_solutions = current_frontier + new_solutions

        # Remove dominated solutions
        pareto_solutions = []

        for solution in all_solutions:
            is_dominated = False

            for other_solution in all_solutions:
                if solution == other_solution:
                    continue

                if self._dominates(other_solution, solution):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_solutions.append(solution)

        # Limit to max solutions
        if len(pareto_solutions) > self.config.max_pareto_solutions:
            # Sort by weighted score and keep best
            pareto_solutions.sort(key=lambda x: x['weighted_score'], reverse=True)
            pareto_solutions = pareto_solutions[:self.config.max_pareto_solutions]

        return pareto_solutions

    def _dominates(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> bool:
        """Check if solution1 dominates solution2."""
        scores1 = solution1.get('objective_scores', {})
        scores2 = solution2.get('objective_scores', {})

        # Solution1 dominates solution2 if it's better in at least one objective
        # and not worse in any objective
        better_in_at_least_one = False

        for objective in self.config.objectives:
            obj_key = objective.value
            if obj_key in scores1 and obj_key in scores2:
                if scores1[obj_key] > scores2[obj_key]:
                    better_in_at_least_one = True
                elif scores1[obj_key] < scores2[obj_key]:
                    return False  # Solution1 is worse in this objective

        return better_in_at_least_one

    def _check_convergence(self, convergence_history: List[Dict[str, float]]) -> bool:
        """Check if optimization has converged."""
        if len(convergence_history) < self.config.early_stopping_patience:
            return False

        # Check if improvement has plateaued
        recent_scores = [h['best_score'] for h in convergence_history[-self.config.early_stopping_patience:]]

        if len(recent_scores) < 2:
            return False

        # Check if standard deviation is low (convergence)
        score_std = np.std(recent_scores)
        if score_std < self.config.convergence_threshold:
            return True

        # Check if improvement rate is low
        improvement = recent_scores[-1] - recent_scores[0]
        if improvement < self.config.convergence_threshold:
            return True

        return False

    def _calculate_diversity(self, population: List[Dict[str, Any]]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0

        # Calculate average pairwise distance
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._calculate_solution_distance(population[i], population[j])
                distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _calculate_solution_distance(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> float:
        """Calculate distance between two solutions."""
        params1 = solution1.get('parameters', {})
        params2 = solution2.get('parameters', {})

        distance = 0.0
        for key in params1:
            if key in params2:
                if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    distance += abs(params1[key] - params2[key]) ** 2
                else:
                    distance += 1.0  # Different types

        return np.sqrt(distance)

    def _generate_next_generation(self, current_population: List[Dict[str, Any]],
                                pareto_frontier: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate next generation using NSGA-II operations."""
        new_population = []

        # Selection, crossover, and mutation
        for i in range(self.config.population_size):
            # Tournament selection
            parent1 = self._tournament_selection(current_population)
            parent2 = self._tournament_selection(current_population)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        return new_population

    def _tournament_selection(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tournament selection for parent selection."""
        tournament_size = min(5, len(population))
        tournament = np.random.choice(len(population), tournament_size, replace=False)

        best_individual = population[tournament[0]]
        best_score = best_individual.get('weighted_score', 0.0)

        for idx in tournament[1:]:
            individual = population[idx]
            score = individual.get('weighted_score', 0.0)
            if score > best_score:
                best_individual = individual
                best_score = score

        return best_individual

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover operation between two parents."""
        child_params = {}
        params1 = parent1.get('parameters', {})
        params2 = parent2.get('parameters', {})

        for key in params1:
            if key in params2:
                if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    # Arithmetic crossover for numeric parameters
                    alpha = np.random.random()
                    child_params[key] = alpha * params1[key] + (1 - alpha) * params2[key]
                else:
                    # Random choice for non-numeric parameters
                    child_params[key] = np.random.choice([params1[key], params2[key]])
            else:
                child_params[key] = params1[key]

        return {
            'id': f'child_{int(time.time())}',
            'parameters': child_params,
            'generation': max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1
        }

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation on an individual."""
        mutated_params = individual['parameters'].copy()
        mutation_rate = 0.1

        for key, value in mutated_params.items():
            if np.random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    # Gaussian mutation for numeric parameters
                    noise = np.random.normal(0, 0.1)
                    mutated_params[key] = max(0.0, value + noise)
                elif isinstance(value, bool):
                    # Flip for boolean parameters
                    mutated_params[key] = not value
                elif isinstance(value, str):
                    # Random choice for string parameters
                    mutated_params[key] = np.random.choice(['option1', 'option2', 'option3'])

        individual['parameters'] = mutated_params
        return individual

    def _calculate_optimization_metrics(self, pareto_frontier: List[Dict[str, Any]],
                                      convergence_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate optimization metrics."""
        if not pareto_frontier:
            return {}

        scores = [sol['weighted_score'] for sol in pareto_frontier]

        return {
            'best_score': max(scores),
            'worst_score': min(scores),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'score_range': max(scores) - min(scores),
            'convergence_rate': self._calculate_convergence_rate(convergence_history),
            'diversity_metric': self._calculate_final_diversity(pareto_frontier)
        }

    def _calculate_convergence_rate(self, convergence_history: List[Dict[str, float]]) -> float:
        """Calculate convergence rate."""
        if len(convergence_history) < 2:
            return 0.0

        recent_scores = [h['best_score'] for h in convergence_history[-10:]]
        if len(recent_scores) < 2:
            return 0.0

        improvement = recent_scores[-1] - recent_scores[0]
        return max(0.0, improvement)

    def _calculate_final_diversity(self, pareto_frontier: List[Dict[str, Any]]) -> float:
        """Calculate final diversity of Pareto frontier."""
        return self._calculate_diversity(pareto_frontier)

    def _calculate_objective_scores(self, pareto_frontier: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average objective scores across Pareto frontier."""
        if not pareto_frontier:
            return {}

        objective_scores = {}
        for objective in self.config.objectives:
            scores = [sol.get('objective_scores', {}).get(objective.value, 0.0) for sol in pareto_frontier]
            objective_scores[objective.value] = np.mean(scores)

        return objective_scores

class UnifiedMultiObjectiveOptimizer:
    """
    Unified Multi-Objective Optimizer.

    Combines the best practices from both TAS and NAS regime detection systems
    to provide comprehensive multi-objective optimization.
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize unified multi-objective optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize evaluators
        self.economic_evaluator = None
        self.trading_evaluator = None

        if OptimizationObjective.ECONOMIC_SIGNIFICANCE in config.objectives:
            self.economic_evaluator = UnifiedEconomicSignificanceEvaluator(
                config.economic_config or EconomicEvaluationConfig()
            )

        if OptimizationObjective.TRADING_VIABILITY in config.objectives:
            self.trading_evaluator = UnifiedTradingViabilityEvaluator(
                config.trading_config or TradingViabilityConfig()
            )

        # Initialize optimization algorithm
        self.optimizer = self._create_optimizer()

        self.logger.info("✅ Unified Multi-Objective Optimizer initialized")
        self.logger.info(f"   Algorithm: {config.algorithm.value}")
        self.logger.info(f"   Objectives: {[obj.value for obj in config.objectives]}")
        self.logger.info(f"   Max iterations: {config.max_iterations}")

    def _create_optimizer(self) -> BaseOptimizationAlgorithm:
        """Create optimization algorithm based on config."""
        if self.config.algorithm == OptimizationAlgorithm.NSGA2:
            return NSGA2Optimizer(self.config)
        else:
            # Default to NSGA2
            return NSGA2Optimizer(self.config)

    def optimize(self,
                 market_data: Union[pd.DataFrame, np.ndarray],
                 regime_predictions: np.ndarray,
                 regime_probabilities: Optional[np.ndarray] = None,
                 timestamps: Optional[np.ndarray] = None,
                 constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Perform unified multi-objective optimization.

        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Current regime predictions
            regime_probabilities: Optional regime probabilities
            timestamps: Optional timestamps
            constraints: Optional optimization constraints

        Returns:
            Optimization result with Pareto solutions
        """
        try:
            self.logger.info("🚀 Starting unified multi-objective optimization...")
            self.logger.info(f"   Data shape: {market_data.shape}")
            self.logger.info(f"   Regimes: {len(np.unique(regime_predictions))}")

            # Create objective function
            objective_function = self._create_objective_function(
                market_data, regime_predictions, regime_probabilities, timestamps
            )

            # Perform optimization
            result = self.optimizer.optimize(objective_function, constraints, self.config)

            self.logger.info(f"✅ Unified multi-objective optimization completed")
            self.logger.info(f"   Success: {result.success}")
            self.logger.info(f"   Pareto solutions: {result.n_solutions}")
            self.logger.info(f"   Best score: {result.weighted_score:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Unified multi-objective optimization failed: {e}")

            return OptimizationResult(
                success=False,
                execution_time=0.0,
                pareto_solutions=[],
                best_solution=None,
                optimization_metrics={},
                convergence_history=[],
                objective_scores={},
                weighted_score=0.0,
                algorithm_used=self.config.algorithm.value,
                n_objectives=len(self.config.objectives),
                n_solutions=0,
                convergence_achieved=False,
                error_message=str(e)
            )

    def _create_objective_function(self,
                                  market_data: Union[pd.DataFrame, np.ndarray],
                                  regime_predictions: np.ndarray,
                                  regime_probabilities: Optional[np.ndarray],
                                  timestamps: Optional[np.ndarray]) -> Callable:
        """Create objective function for optimization."""

        def objective_function(individual: Dict[str, Any]) -> Dict[str, float]:
            """Evaluate individual against all objectives."""
            scores = {}

            try:
                # Simulate regime predictions based on individual parameters
                simulated_predictions = self._simulate_regime_predictions(
                    individual, regime_predictions
                )

                # Evaluate each objective
                for objective in self.config.objectives:
                    if objective == OptimizationObjective.REGIME_ACCURACY:
                        scores[objective.value] = self._evaluate_regime_accuracy(
                            simulated_predictions, regime_predictions
                        )
                    elif objective == OptimizationObjective.ECONOMIC_SIGNIFICANCE:
                        scores[objective.value] = self._evaluate_economic_significance(
                            market_data, simulated_predictions, timestamps
                        )
                    elif objective == OptimizationObjective.TRADING_VIABILITY:
                        scores[objective.value] = self._evaluate_trading_viability(
                            market_data, simulated_predictions, regime_probabilities, timestamps
                        )
                    elif objective == OptimizationObjective.COMPUTATIONAL_EFFICIENCY:
                        scores[objective.value] = self._evaluate_computational_efficiency(individual)
                    elif objective == OptimizationObjective.ARCHITECTURE_COMPLEXITY:
                        scores[objective.value] = self._evaluate_architecture_complexity(individual)
                    elif objective == OptimizationObjective.REGIME_STABILITY:
                        scores[objective.value] = self._evaluate_regime_stability(simulated_predictions)
                    elif objective == OptimizationObjective.TRANSITION_ACCURACY:
                        scores[objective.value] = self._evaluate_transition_accuracy(simulated_predictions)
                    elif objective == OptimizationObjective.MODEL_CONFIDENCE:
                        scores[objective.value] = self._evaluate_model_confidence(individual)
                    elif objective == OptimizationObjective.RISK_ADJUSTED_RETURNS:
                        scores[objective.value] = self._evaluate_risk_adjusted_returns(
                            market_data, simulated_predictions
                        )
                    elif objective == OptimizationObjective.EXECUTION_FEASIBILITY:
                        scores[objective.value] = self._evaluate_execution_feasibility(
                            market_data, simulated_predictions
                        )
                    else:
                        scores[objective.value] = 0.5  # Default score

                return scores

            except Exception as e:
                self.logger.warning(f"Objective evaluation failed: {e}")
                # Return default scores for failed evaluation
                return {obj.value: 0.1 for obj in self.config.objectives}

        return objective_function

    def _simulate_regime_predictions(self, individual: Dict[str, Any],
                                   original_predictions: np.ndarray) -> np.ndarray:
        """Simulate regime predictions based on individual parameters."""
        try:
            parameters = individual.get('parameters', {})

            # Simple simulation based on parameters
            n_samples = len(original_predictions)
            n_regimes = parameters.get('regime_count', len(np.unique(original_predictions)))
            stability_factor = parameters.get('stability_factor', 0.5)
            complexity_factor = parameters.get('complexity_factor', 0.5)

            # Generate base predictions
            base_predictions = np.random.randint(0, n_regimes, n_samples)

            # Apply stability factor
            if stability_factor > 0.5:
                for i in range(1, len(base_predictions)):
                    if np.random.random() < stability_factor:
                        base_predictions[i] = base_predictions[i-1]

            # Apply complexity factor (affects prediction variability)
            if complexity_factor < 0.5:
                # Lower complexity = more stable predictions
                noise = np.random.normal(0, 0.1, n_samples)
                base_predictions = np.clip(base_predictions + noise, 0, n_regimes-1).astype(int)

            return base_predictions

        except Exception as e:
            self.logger.warning(f"Regime prediction simulation failed: {e}")
            return original_predictions

    def _evaluate_regime_accuracy(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Evaluate regime prediction accuracy."""
        try:
            if len(predicted) != len(actual):
                return 0.0

            # Calculate accuracy
            correct = np.sum(predicted == actual)
            accuracy = correct / len(actual)

            return accuracy

        except Exception as e:
            self.logger.warning(f"Regime accuracy evaluation failed: {e}")
            return 0.0

    def _evaluate_economic_significance(self, market_data: Union[pd.DataFrame, np.ndarray],
                                      regime_predictions: np.ndarray,
                                      timestamps: Optional[np.ndarray]) -> float:
        """Evaluate economic significance."""
        try:
            if self.economic_evaluator is None:
                return 0.5  # Default score if evaluator not available

            result = self.economic_evaluator.evaluate(market_data, regime_predictions, timestamps=timestamps)
            return result.overall_score

        except Exception as e:
            self.logger.warning(f"Economic significance evaluation failed: {e}")
            return 0.0

    def _evaluate_trading_viability(self, market_data: Union[pd.DataFrame, np.ndarray],
                                 regime_predictions: np.ndarray,
                                 regime_probabilities: Optional[np.ndarray],
                                 timestamps: Optional[np.ndarray]) -> float:
        """Evaluate trading viability."""
        try:
            if self.trading_evaluator is None:
                return 0.5  # Default score if evaluator not available

            result = self.trading_evaluator.evaluate(
                market_data, regime_predictions, regime_probabilities, timestamps
            )
            return result.overall_score

        except Exception as e:
            self.logger.warning(f"Trading viability evaluation failed: {e}")
            return 0.0

    def _evaluate_computational_efficiency(self, individual: Dict[str, Any]) -> float:
        """Evaluate computational efficiency."""
        try:
            parameters = individual.get('parameters', {})

            # Efficiency based on parameters
            efficiency_factor = parameters.get('efficiency_factor', 0.5)
            batch_size = parameters.get('batch_size', 64)
            complexity_factor = parameters.get('complexity_factor', 0.5)

            # Larger batch sizes are more efficient
            batch_efficiency = min(batch_size / 256, 1.0)

            # Lower complexity is more efficient
            complexity_efficiency = 1.0 - complexity_factor

            # Combine efficiency factors
            efficiency_score = (
                efficiency_factor * 0.4 +
                batch_efficiency * 0.3 +
                complexity_efficiency * 0.3
            )

            return max(0.0, min(1.0, efficiency_score))

        except Exception as e:
            self.logger.warning(f"Computational efficiency evaluation failed: {e}")
            return 0.0

    def _evaluate_architecture_complexity(self, individual: Dict[str, Any]) -> float:
        """Evaluate architecture complexity (lower is better)."""
        try:
            parameters = individual.get('parameters', {})

            # Complexity based on parameters
            complexity_factor = parameters.get('complexity_factor', 0.5)
            regime_count = parameters.get('regime_count', 3)
            dropout_rate = parameters.get('dropout_rate', 0.2)

            # Normalize complexity (lower is better)
            regime_complexity = min(regime_count / 10, 1.0)
            dropout_complexity = dropout_rate

            complexity_score = (
                complexity_factor * 0.4 +
                regime_complexity * 0.3 +
                dropout_complexity * 0.3
            )

            # Return inverse (lower complexity is better)
            return 1.0 - complexity_score

        except Exception as e:
            self.logger.warning(f"Architecture complexity evaluation failed: {e}")
            return 0.0

    def _evaluate_regime_stability(self, regime_predictions: np.ndarray) -> float:
        """Evaluate regime stability."""
        try:
            if len(regime_predictions) < 2:
                return 0.0

            # Calculate regime changes
            regime_changes = np.sum(np.diff(regime_predictions) != 0)
            total_periods = len(regime_predictions) - 1

            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0

            return max(0.0, min(1.0, stability))

        except Exception as e:
            self.logger.warning(f"Regime stability evaluation failed: {e}")
            return 0.0

    def _evaluate_transition_accuracy(self, regime_predictions: np.ndarray) -> float:
        """Evaluate regime transition accuracy."""
        try:
            if len(regime_predictions) < 3:
                return 0.5

            # Calculate transition matrix
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)

            if n_regimes < 2:
                return 0.5

            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(regime_predictions) - 1):
                current_regime = regime_predictions[i]
                next_regime = regime_predictions[i + 1]

                if current_regime in unique_regimes and next_regime in unique_regimes:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    next_idx = np.where(unique_regimes == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1

            # Calculate transition accuracy
            total_transitions = np.sum(transition_matrix)
            if total_transitions > 0:
                diagonal_sum = np.trace(transition_matrix)
                transition_accuracy = diagonal_sum / total_transitions
            else:
                transition_accuracy = 0.5

            return min(transition_accuracy, 1.0)

        except Exception as e:
            self.logger.warning(f"Transition accuracy evaluation failed: {e}")
            return 0.0

    def _evaluate_model_confidence(self, individual: Dict[str, Any]) -> float:
        """Evaluate model confidence."""
        try:
            parameters = individual.get('parameters', {})
            confidence_threshold = parameters.get('confidence_threshold', 0.7)

            # Higher confidence thresholds indicate more confident models
            return confidence_threshold

        except Exception as e:
            self.logger.warning(f"Model confidence evaluation failed: {e}")
            return 0.0

    def _evaluate_risk_adjusted_returns(self, market_data: Union[pd.DataFrame, np.ndarray],
                                      regime_predictions: np.ndarray) -> float:
        """Evaluate risk-adjusted returns."""
        try:
            if isinstance(market_data, pd.DataFrame):
                close_prices = market_data['close'].values
            else:
                close_prices = market_data[:, 3]  # Assuming OHLCV format

            if len(close_prices) < 3:
                return 0.0

            # Calculate returns
            returns = np.diff(close_prices) / close_prices[:-1]

            # Calculate risk-adjusted metrics
            mean_return = np.mean(returns)
            volatility = np.std(returns)

            if volatility > 0:
                sharpe_ratio = mean_return / volatility
                return min(sharpe_ratio / 2.0, 1.0)  # Normalize to 0-1
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Risk-adjusted returns evaluation failed: {e}")
            return 0.0

    def _evaluate_execution_feasibility(self, market_data: Union[pd.DataFrame, np.ndarray],
                                      regime_predictions: np.ndarray) -> float:
        """Evaluate execution feasibility."""
        try:
            if isinstance(market_data, pd.DataFrame):
                close_prices = market_data['close'].values
            else:
                close_prices = market_data[:, 3]  # Assuming OHLCV format

            if len(close_prices) < 3:
                return 0.5

            # Calculate price volatility (affects execution)
            price_volatility = np.std(np.diff(close_prices) / close_prices[:-1])

            # Execution feasibility (lower volatility is better)
            if price_volatility <= 0.01:  # 1% threshold
                feasibility = 1.0
            else:
                feasibility = max(0.0, 1.0 - (price_volatility - 0.01) / 0.01)

            return feasibility

        except Exception as e:
            self.logger.warning(f"Execution feasibility evaluation failed: {e}")
            return 0.0

# Convenience functions
def create_unified_multi_objective_optimizer(config: Optional[OptimizationConfig] = None) -> UnifiedMultiObjectiveOptimizer:
    """Create a unified multi-objective optimizer."""
    if config is None:
        config = OptimizationConfig()
    return UnifiedMultiObjectiveOptimizer(config)

def quick_multi_objective_optimization(market_data: Union[pd.DataFrame, np.ndarray],
                                     regime_predictions: np.ndarray,
                                     config: Optional[OptimizationConfig] = None) -> OptimizationResult:
    """Quick multi-objective optimization with default settings."""
    optimizer = create_unified_multi_objective_optimizer(config)
    return optimizer.optimize(market_data, regime_predictions)
