"""
Multi-Objective Optimization for Architecture Search

This module implements multi-objective optimization for neural and tree architectures,
focusing on trading-specific objectives like Sharpe ratio, maximum drawdown, win rate,
and profit factor.
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
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Import existing Pareto optimization
from ....backtesting.pareto import ParetoOptimizer

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    PERFORMANCE = "performance"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    VOLATILITY = "volatility"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    COMPLEXITY = "complexity"
    LATENCY = "latency"


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    objectives: List[ObjectiveType] = field(default_factory=lambda: [
        ObjectiveType.PERFORMANCE,
        ObjectiveType.SHARPE_RATIO,
        ObjectiveType.MAX_DRAWDOWN,
        ObjectiveType.WIN_RATE
    ])

    # Objective weights
    weights: Dict[ObjectiveType, float] = field(default_factory=lambda: {
        ObjectiveType.PERFORMANCE: 1.0,
        ObjectiveType.SHARPE_RATIO: 0.8,
        ObjectiveType.MAX_DRAWDOWN: 0.6,
        ObjectiveType.WIN_RATE: 0.7,
        ObjectiveType.PROFIT_FACTOR: 0.5,
        ObjectiveType.VOLATILITY: 0.4,
        ObjectiveType.CALMAR_RATIO: 0.6,
        ObjectiveType.SORTINO_RATIO: 0.5,
        ObjectiveType.COMPLEXITY: 0.3,
        ObjectiveType.LATENCY: 0.2
    })

    # Pareto optimization settings
    use_pareto_optimization: bool = True
    pareto_population_size: int = 100
    pareto_max_generations: int = 50

    # Constraints
    max_complexity: float = 1.0
    max_latency_ms: float = 100.0
    min_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.3

    # Optimization settings
    scalarization_method: str = "weighted_sum"  # or "chebyshev", "achievement_scalarizing"
    constraint_handling: str = "penalty"  # or "feasibility_rules"


@dataclass
class ObjectiveValue:
    """Value for a single objective."""
    objective_type: ObjectiveType
    value: float
    normalized_value: float
    is_constrained: bool = False
    constraint_violation: float = 0.0


@dataclass
class MultiObjectiveResult:
    """Result from multi-objective optimization."""
    pareto_frontier: List[Dict[str, Any]]
    best_architecture: Dict[str, Any]
    objective_values: Dict[str, ObjectiveValue]
    scalarized_score: float
    dominance_rank: int
    crowding_distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingMultiObjectiveOptimizer:
    """
    Multi-objective optimizer for trading architectures.

    Optimizes neural and tree architectures using trading-specific objectives
    including Sharpe ratio, maximum drawdown, win rate, and profit factor.
    """

    def __init__(self, config: MultiObjectiveConfig):
        """Initialize the multi-objective optimizer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize Pareto optimizer
        if self.config.use_pareto_optimization:
            self.pareto_optimizer = ParetoOptimizer(
                population_size=self.config.pareto_population_size,
                max_generations=self.config.pareto_max_generations,
                crossover_rate=0.8,
                mutation_rate=0.1
            )

        # Objective scalers for normalization
        self.scalers = {obj: StandardScaler() for obj in self.config.objectives}

        self.logger.info("✅ Trading Multi-Objective Optimizer initialized")
        self.logger.info(f"   Objectives: {[obj.value for obj in config.objectives]}")
        self.logger.info(f"   Pareto Optimization: {config.use_pareto_optimization}")

    def optimize(self,
                 architectures: List[Dict[str, Any]],
                 objective_evaluator: Callable[[Dict[str, Any]], Dict[ObjectiveType, float]],
                 constraint_checker: Callable[[Dict[str, Any]], bool]) -> MultiObjectiveResult:
        """Perform multi-objective optimization on architectures."""
        start_time = time.time()
        self.logger.info("🚀 Starting Multi-Objective Optimization...")

        try:
            # Evaluate all architectures
            evaluated_architectures = []
            for arch in architectures:
                if constraint_checker(arch):
                    objectives = objective_evaluator(arch)
                    evaluated_architectures.append({
                        'architecture': arch,
                        'objectives': objectives
                    })

            if not evaluated_architectures:
                raise ValueError("No valid architectures found for optimization")

            # Normalize objectives
            normalized_architectures = self._normalize_objectives(evaluated_architectures)

            # Perform Pareto optimization if enabled
            if self.config.use_pareto_optimization:
                pareto_frontier = self._pareto_optimization(normalized_architectures)
            else:
                pareto_frontier = normalized_architectures

            # Find best architecture using scalarization
            best_result = self._find_best_architecture(pareto_frontier)

            execution_time = time.time() - start_time

            result = MultiObjectiveResult(
                pareto_frontier=pareto_frontier,
                best_architecture=best_result['architecture'],
                objective_values=best_result['objective_values'],
                scalarized_score=best_result['scalarized_score'],
                dominance_rank=best_result['dominance_rank'],
                crowding_distance=best_result['crowding_distance'],
                metadata={
                    'n_architectures': len(architectures),
                    'n_valid_architectures': len(evaluated_architectures),
                    'n_pareto_optimal': len(pareto_frontier),
                    'execution_time': execution_time,
                    'scalarization_method': self.config.scalarization_method,
                    'constraint_handling': self.config.constraint_handling
                }
            )

            self.logger.info(f"✅ Multi-Objective Optimization completed in {execution_time:.2f}s")
            self.logger.info(f"   Pareto Front Size: {len(pareto_frontier)}")
            self.logger.info(f"   Best Scalarized Score: {result.scalarized_score:.4f}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Multi-Objective Optimization failed: {e}")

            return MultiObjectiveResult(
                pareto_frontier=[],
                best_architecture={},
                objective_values={},
                scalarized_score=0.0,
                dominance_rank=0,
                crowding_distance=0.0,
                metadata={'error': str(e), 'execution_time': execution_time}
            )

    def _normalize_objectives(self, evaluated_architectures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize objective values to [0,1] range."""
        try:
            # Extract objective values for each architecture
            objective_matrices = {obj: [] for obj in self.config.objectives}

            for arch in evaluated_architectures:
                for obj_type, value in arch['objectives'].items():
                    if obj_type in objective_matrices:
                        objective_matrices[obj_type].append(value)

            # Normalize each objective
            normalized_architectures = []
            for arch in evaluated_architectures:
                normalized_objectives = {}

                for obj_type in self.config.objectives:
                    if obj_type in arch['objectives']:
                        value = arch['objectives'][obj_type]

                        # Handle different objective types
                        if obj_type in [ObjectiveType.MAX_DRAWDOWN, ObjectiveType.VOLATILITY]:
                            # Lower values are better (inverted normalization)
                            if len(objective_matrices[obj_type]) > 1:
                                max_val = max(objective_matrices[obj_type])
                                min_val = min(objective_matrices[obj_type])
                                if max_val != min_val:
                                    normalized_value = (max_val - value) / (max_val - min_val)
                                else:
                                    normalized_value = 1.0
                            else:
                                normalized_value = 1.0
                        else:
                            # Higher values are better (standard normalization)
                            if len(objective_matrices[obj_type]) > 1:
                                max_val = max(objective_matrices[obj_type])
                                min_val = min(objective_matrices[obj_type])
                                if max_val != min_val:
                                    normalized_value = (value - min_val) / (max_val - min_val)
                                else:
                                    normalized_value = 1.0
                            else:
                                normalized_value = 1.0

                        normalized_objectives[obj_type] = ObjectiveValue(
                            objective_type=obj_type,
                            value=value,
                            normalized_value=max(0.0, min(1.0, normalized_value))
                        )

                normalized_architectures.append({
                    'architecture': arch['architecture'],
                    'objectives': normalized_objectives,
                    'raw_objectives': arch['objectives']
                })

            return normalized_architectures

        except Exception as e:
            self.logger.error(f"❌ Objective normalization failed: {e}")
            return evaluated_architectures

    def _pareto_optimization(self, normalized_architectures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform Pareto optimization to find non-dominated solutions."""
        try:
            # Extract objective values for Pareto optimization
            objective_values = []
            architecture_indices = []

            for i, arch in enumerate(normalized_architectures):
                values = [obj.value.normalized_value for obj in arch['objectives'].values()]
                objective_values.append(values)
                architecture_indices.append(i)

            # Use existing Pareto optimizer
            if self.pareto_optimizer:
                pareto_indices = self.pareto_optimizer.find_pareto_frontier(
                    np.array(objective_values)
                )

                # Return Pareto-optimal architectures
                pareto_frontier = []
                for idx in pareto_indices:
                    pareto_frontier.append(normalized_architectures[architecture_indices[idx]])

                return pareto_frontier
            else:
                # Fallback: return all architectures
                return normalized_architectures

        except Exception as e:
            self.logger.error(f"❌ Pareto optimization failed: {e}")
            return normalized_architectures

    def _find_best_architecture(self, pareto_frontier: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best architecture using scalarization."""
        best_score = -np.inf
        best_architecture = None
        best_objective_values = {}
        best_dominance_rank = 0
        best_crowding_distance = 0

        for arch in pareto_frontier:
            # Calculate scalarized score using configured method
            if self.config.scalarization_method == "weighted_sum":
                score = self._weighted_sum_scalarization(arch['objectives'])
            elif self.config.scalarization_method == "chebyshev":
                score = self._chebyshev_scalarization(arch['objectives'])
            else:
                score = self._weighted_sum_scalarization(arch['objectives'])

            # Apply constraint penalties
            penalty = self._calculate_constraint_penalty(arch['architecture'])
            score -= penalty

            # Track additional metrics
            dominance_rank = self._calculate_dominance_rank(arch, pareto_frontier)
            crowding_distance = self._calculate_crowding_distance(arch, pareto_frontier)

            if score > best_score:
                best_score = score
                best_architecture = arch['architecture']
                best_objective_values = arch['objectives']
                best_dominance_rank = dominance_rank
                best_crowding_distance = crowding_distance

        return {
            'architecture': best_architecture,
            'objective_values': best_objective_values,
            'scalarized_score': best_score,
            'dominance_rank': best_dominance_rank,
            'crowding_distance': best_crowding_distance
        }

    def _weighted_sum_scalarization(self, objectives: Dict[ObjectiveType, ObjectiveValue]) -> float:
        """Calculate weighted sum scalarization."""
        total_score = 0.0
        total_weight = 0.0

        for obj_type, obj_value in objectives.items():
            if obj_type in self.config.weights:
                weight = self.config.weights[obj_type]
                total_score += weight * obj_value.normalized_value
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _chebyshev_scalarization(self, objectives: Dict[ObjectiveType, ObjectiveValue]) -> float:
        """Calculate Chebyshev scalarization."""
        max_deviation = 0.0
        total_weight = 0.0

        for obj_type, obj_value in objectives.items():
            if obj_type in self.config.weights:
                weight = self.config.weights[obj_type]
                # Chebyshev uses maximum weighted deviation from ideal
                deviation = weight * (1.0 - obj_value.normalized_value)
                max_deviation = max(max_deviation, deviation)
                total_weight += weight

        return -max_deviation  # Negative because lower deviation is better

    def _calculate_constraint_penalty(self, architecture: Dict[str, Any]) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0

        # Complexity constraint
        complexity = self._calculate_architecture_complexity(architecture)
        if complexity > self.config.max_complexity:
            penalty += (complexity - self.config.max_complexity) * 10.0

        # Latency constraint
        latency = self._estimate_architecture_latency(architecture)
        if latency > self.config.max_latency_ms:
            penalty += (latency - self.config.max_latency_ms) * 0.1

        # Trading-specific constraints
        if ObjectiveType.SHARPE_RATIO in self.config.objectives:
            estimated_sharpe = self._estimate_sharpe_ratio(architecture)
            if estimated_sharpe < self.config.min_sharpe_ratio:
                penalty += (self.config.min_sharpe_ratio - estimated_sharpe) * 5.0

        return penalty

    def _calculate_architecture_complexity(self, architecture: Dict[str, Any]) -> float:
        """Calculate architecture complexity (0-1 scale)."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)
            total_params = sum(layer.get('hidden_size', 100) for layer in layers)

            # Normalize complexity
            normalized_layers = min(n_layers / 10.0, 1.0)
            normalized_params = min(total_params / 100000.0, 1.0)

            return 0.6 * normalized_layers + 0.4 * normalized_params

        except Exception:
            return 0.5

    def _estimate_architecture_latency(self, architecture: Dict[str, Any]) -> float:
        """Estimate architecture inference latency in milliseconds."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)
            total_params = sum(layer.get('hidden_size', 100) for layer in layers)

            # Simple latency estimation (this would be more sophisticated in practice)
            base_latency = 10.0  # Base latency for simple architecture
            layer_penalty = n_layers * 2.0
            param_penalty = total_params / 10000.0

            return base_latency + layer_penalty + param_penalty

        except Exception:
            return 50.0

    def _estimate_sharpe_ratio(self, architecture: Dict[str, Any]) -> float:
        """Estimate Sharpe ratio for architecture."""
        try:
            # This would use historical data and architecture characteristics
            # For now, return a mock value based on architecture properties
            layers = architecture.get('layers', [])
            complexity = self._calculate_architecture_complexity(architecture)

            # Higher complexity architectures might have better Sharpe ratios
            # but with diminishing returns
            base_sharpe = 0.8
            complexity_bonus = min(complexity * 0.5, 0.4)

            return base_sharpe + complexity_bonus

        except Exception:
            return 0.5

    def _calculate_dominance_rank(self, architecture: Dict[str, Any],
                                pareto_frontier: List[Dict[str, Any]]) -> int:
        """Calculate dominance rank for architecture."""
        # Simplified dominance ranking
        # In practice, this would use NSGA-II or similar algorithm
        return 1  # Assume Pareto-optimal

    def _calculate_crowding_distance(self, architecture: Dict[str, Any],
                                   pareto_frontier: List[Dict[str, Any]]) -> float:
        """Calculate crowding distance for architecture."""
        # Simplified crowding distance
        # In practice, this would calculate actual crowding distance
        n_frontier = len(pareto_frontier)
        return 1.0 / n_frontier if n_frontier > 0 else 0.0

    def evaluate_trading_objectives(self, architecture: Dict[str, Any],
                                  market_data: pd.DataFrame) -> Dict[ObjectiveType, float]:
        """Evaluate trading-specific objectives for an architecture."""
        try:
            objectives = {}

            # Simulate trading performance
            trading_metrics = self._simulate_trading_performance(architecture, market_data)

            # Performance (accuracy)
            objectives[ObjectiveType.PERFORMANCE] = trading_metrics.get('accuracy', 0.5)

            # Sharpe ratio
            objectives[ObjectiveType.SHARPE_RATIO] = trading_metrics.get('sharpe_ratio', 0.8)

            # Maximum drawdown (negative because lower is better)
            objectives[ObjectiveType.MAX_DRAWDOWN] = trading_metrics.get('max_drawdown', -0.15)

            # Win rate
            objectives[ObjectiveType.WIN_RATE] = trading_metrics.get('win_rate', 0.55)

            # Profit factor
            objectives[ObjectiveType.PROFIT_FACTOR] = trading_metrics.get('profit_factor', 1.2)

            # Volatility
            objectives[ObjectiveType.VOLATILITY] = trading_metrics.get('volatility', 0.02)

            # Calmar ratio (annual return / max drawdown)
            annual_return = trading_metrics.get('annual_return', 0.12)
            max_drawdown = abs(trading_metrics.get('max_drawdown', -0.15))
            objectives[ObjectiveType.CALMAR_RATIO] = annual_return / max_drawdown if max_drawdown > 0 else 0.0

            # Sortino ratio (annual return / downside deviation)
            downside_deviation = trading_metrics.get('downside_deviation', 0.02)
            objectives[ObjectiveType.SORTINO_RATIO] = annual_return / downside_deviation if downside_deviation > 0 else 0.0

            # Complexity
            objectives[ObjectiveType.COMPLEXITY] = self._calculate_architecture_complexity(architecture)

            # Latency
            objectives[ObjectiveType.LATENCY] = self._estimate_architecture_latency(architecture)

            return objectives

        except Exception as e:
            self.logger.error(f"❌ Trading objective evaluation failed: {e}")
            return {obj: 0.5 for obj in self.config.objectives}

    def _simulate_trading_performance(self, architecture: Dict[str, Any],
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """Simulate trading performance for architecture evaluation."""
        try:
            # This is a simplified simulation
            # In practice, this would involve backtesting the architecture

            n_samples = len(market_data)

            # Simulate trading metrics based on architecture characteristics
            complexity = self._calculate_architecture_complexity(architecture)
            layers = architecture.get('layers', [])
            n_layers = len(layers)

            # Base metrics
            accuracy = 0.5 + np.random.normal(0, 0.1)  # Random around 50%
            accuracy = max(0.0, min(1.0, accuracy))

            # Architecture-specific adjustments
            if n_layers > 5:
                accuracy += 0.05  # Deeper networks might be more accurate
            if complexity > 0.7:
                accuracy -= 0.03  # Very complex architectures might overfit

            # Sharpe ratio
            sharpe_ratio = 0.8 + np.random.normal(0, 0.3)
            sharpe_ratio = max(0.0, sharpe_ratio)

            # Maximum drawdown
            max_drawdown = -0.15 + np.random.normal(0, 0.05)
            max_drawdown = min(0.0, max_drawdown)

            # Win rate
            win_rate = 0.55 + np.random.normal(0, 0.1)
            win_rate = max(0.4, min(0.7, win_rate))

            # Profit factor
            profit_factor = 1.2 + np.random.normal(0, 0.3)
            profit_factor = max(1.0, profit_factor)

            # Volatility
            volatility = 0.02 + np.random.normal(0, 0.01)
            volatility = max(0.005, volatility)

            # Annual return
            annual_return = 0.12 + np.random.normal(0, 0.05)
            annual_return = max(-0.1, annual_return)

            return {
                'accuracy': accuracy,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'volatility': volatility,
                'annual_return': annual_return,
                'downside_deviation': volatility * 0.7  # Simplified
            }

        except Exception as e:
            self.logger.warning(f"Trading simulation failed: {e}")
            return {
                'accuracy': 0.5,
                'sharpe_ratio': 0.8,
                'max_drawdown': -0.15,
                'win_rate': 0.55,
                'profit_factor': 1.2,
                'volatility': 0.02,
                'annual_return': 0.12,
                'downside_deviation': 0.015
            }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization configuration and results."""
        return {
            'objectives': [obj.value for obj in self.config.objectives],
            'weights': {obj.value: weight for obj, weight in self.config.weights.items()},
            'pareto_optimization_enabled': self.config.use_pareto_optimization,
            'scalarization_method': self.config.scalarization_method,
            'constraints': {
                'max_complexity': self.config.max_complexity,
                'max_latency_ms': self.config.max_latency_ms,
                'min_sharpe_ratio': self.config.min_sharpe_ratio,
                'max_drawdown': self.config.max_drawdown
            }
        }


def create_multi_objective_optimizer(config: MultiObjectiveConfig) -> TradingMultiObjectiveOptimizer:
    """Create a multi-objective optimizer instance."""
    return TradingMultiObjectiveOptimizer(config)


def quick_multi_objective_search(architectures: List[Dict[str, Any]],
                                objective_evaluator: Callable,
                                config: Optional[MultiObjectiveConfig] = None) -> MultiObjectiveResult:
    """Quick multi-objective optimization with default settings."""
    if config is None:
        config = MultiObjectiveConfig()

    optimizer = TradingMultiObjectiveOptimizer(config)

    def constraint_checker(arch):
        return True  # Accept all architectures in quick mode

    return optimizer.optimize(architectures, objective_evaluator, constraint_checker)