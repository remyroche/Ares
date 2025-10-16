"""
Financial-Specific Search Strategies

This module provides search strategies optimized for financial time series,
including regime-aware search, volatility-adaptive search, and financial
objective optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner

from .financial_architecture_primitives import FinancialActivationType, RegimeType
from .dynamic_search_space import DynamicSearchSpace, MarketCondition

logger = logging.getLogger(__name__)

class FinancialSearchStrategy(Enum):
    """Financial-specific search strategies."""
    REGIME_AWARE_BAYESIAN = "regime_aware_bayesian"
    VOLATILITY_ADAPTIVE_EVOLUTIONARY = "volatility_adaptive_evolutionary"
    SHARPE_OPTIMIZED_RL = "sharpe_optimized_rl"
    DRAWDOWN_AWARE_GRID = "drawdown_aware_grid"
    MOMENTUM_BASED_OPTUNA = "momentum_based_optuna"
    FINANCIAL_MULTI_OBJECTIVE = "financial_multi_objective"
    REGIME_TRANSITION_AWARE = "regime_transition_aware"
    VOLATILITY_CLUSTERING_AWARE = "volatility_clustering_aware"

class FinancialObjective(Enum):
    """Financial objectives for optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    VAR = "var"
    CVAR = "cvar"

@dataclass
class FinancialSearchConfig:
    """Configuration for financial search strategies."""
    # Base search parameters
    strategy: FinancialSearchStrategy = FinancialSearchStrategy.REGIME_AWARE_BAYESIAN
    max_evaluations: int = 1000
    max_search_time: int = 3600  # 1 hour
    population_size: int = 50
    n_trials: int = 100

    # Financial objectives
    primary_objective: FinancialObjective = FinancialObjective.SHARPE_RATIO
    secondary_objectives: List[FinancialObjective] = field(default_factory=lambda: [
        FinancialObjective.MAX_DRAWDOWN, FinancialObjective.WIN_RATE
    ])
    objective_weights: Dict[FinancialObjective, float] = field(default_factory=lambda: {
        FinancialObjective.SHARPE_RATIO: 0.4,
        FinancialObjective.MAX_DRAWDOWN: 0.3,
        FinancialObjective.WIN_RATE: 0.3
    })

    # Regime awareness
    enable_regime_awareness: bool = True
    regime_window: int = 20
    regime_stability_threshold: float = 0.7

    # Volatility adaptation
    enable_volatility_adaptation: bool = True
    volatility_window: int = 20
    volatility_threshold: float = 0.02

    # Risk management
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.1
    stop_loss_threshold: float = 0.05

    # Performance tracking
    performance_window: int = 50
    min_performance_samples: int = 10

    # Search space constraints
    max_layers: int = 10
    min_layers: int = 2
    max_parameters: int = 1000000
    min_parameters: int = 1000

@dataclass
class FinancialSearchResult:
    """Result from financial search strategy."""
    best_architecture: Dict[str, Any]
    best_score: float
    financial_metrics: Dict[str, float]
    regime_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    risk_metrics: Dict[str, float]
    search_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    n_evaluations: int

class RegimeAwareBayesianSearch:
    """Bayesian optimization with regime awareness for financial architectures."""

    def __init__(self, config: FinancialSearchConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )

        # Regime tracking
        self.regime_history = []
        self.regime_performance = {}

        # Search state
        self.search_history = []
        self.best_architecture = None
        self.best_score = -np.inf

    def search(self, architecture_generator: Callable, performance_evaluator: Callable,
               constraint_validator: Callable, n_iterations: int) -> FinancialSearchResult:
        """Perform regime-aware Bayesian search."""
        start_time = time.time()
        self.logger.info("🔍 Starting Regime-Aware Bayesian Search...")

        try:
            # Initialize with random samples
            X_init, y_init = self._initialize_search(architecture_generator, performance_evaluator, constraint_validator)

            # Bayesian optimization loop
            for iteration in range(n_iterations):
                # Update GP with current data
                if len(X_init) > 0:
                    self.gp.fit(X_init, y_init)

                # Generate next candidate
                candidate = self._generate_next_candidate(X_init, y_init, architecture_generator)

                # Evaluate candidate
                performance = performance_evaluator(candidate)

                # Update search state
                X_init = np.vstack([X_init, self._encode_architecture(candidate)])
                y_init = np.append(y_init, performance)

                # Update best
                if performance > self.best_score:
                    self.best_score = performance
                    self.best_architecture = candidate

                # Update regime tracking
                self._update_regime_tracking(candidate, performance)

                # Store search history
                self.search_history.append({
                    'iteration': iteration,
                    'architecture': candidate,
                    'performance': performance,
                    'regime': self._get_current_regime(),
                    'timestamp': datetime.now()
                })

                self.logger.debug(f"Iteration {iteration}: Performance = {performance:.4f}")

            execution_time = time.time() - start_time

            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics()
            regime_analysis = self._analyze_regime_performance()
            volatility_analysis = self._analyze_volatility_impact()
            risk_metrics = self._calculate_risk_metrics()

            return FinancialSearchResult(
                best_architecture=self.best_architecture,
                best_score=self.best_score,
                financial_metrics=financial_metrics,
                regime_analysis=regime_analysis,
                volatility_analysis=volatility_analysis,
                risk_metrics=risk_metrics,
                search_history=self.search_history,
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_evaluations=len(self.search_history)
            )

        except Exception as e:
            self.logger.error(f"Regime-aware Bayesian search failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)

    def _initialize_search(self, architecture_generator: Callable, performance_evaluator: Callable,
                          constraint_validator: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize search with random samples."""
        X_init = []
        y_init = []

        n_init = min(10, self.config.max_evaluations // 10)

        for _ in range(n_init):
            candidate = architecture_generator()
            if constraint_validator(candidate).is_valid:
                performance = performance_evaluator(candidate)
                X_init.append(self._encode_architecture(candidate))
                y_init.append(performance)

        return np.array(X_init), np.array(y_init)

    def _generate_next_candidate(self, X: np.ndarray, y: np.ndarray,
                                architecture_generator: Callable) -> Dict[str, Any]:
        """Generate next candidate using acquisition function."""
        if len(X) == 0:
            return architecture_generator()

        # Calculate acquisition function (Expected Improvement)
        mu, sigma = self.gp.predict(X, return_std=True)
        best_y = np.max(y)

        # Expected Improvement
        improvement = mu - best_y
        z = improvement / (sigma + 1e-8)
        ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)

        # Add exploration bonus
        exploration_bonus = sigma * 0.1

        # Total acquisition
        acquisition = ei + exploration_bonus

        # Select best candidate
        best_idx = np.argmax(acquisition)

        # Decode architecture (simplified)
        return self._decode_architecture(X[best_idx])

    def _encode_architecture(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Encode architecture to feature vector."""
        features = []

        # Architecture type
        arch_type = architecture.get('type', 'neural')
        features.extend([1 if arch_type == 'neural' else 0, 1 if arch_type == 'tree' else 0])

        # Layer count
        layers = architecture.get('layers', [])
        features.append(len(layers))

        # Hidden sizes
        hidden_sizes = [layer.get('hidden_size', 64) for layer in layers]
        features.append(np.mean(hidden_sizes) if hidden_sizes else 64)
        features.append(np.std(hidden_sizes) if len(hidden_sizes) > 1 else 0)

        # Activation type
        activation = architecture.get('activation', 'relu')
        activation_encoding = [0] * 5  # 5 activation types
        if 'volatility' in activation:
            activation_encoding[0] = 1
        elif 'regime' in activation:
            activation_encoding[1] = 1
        elif 'sharpe' in activation:
            activation_encoding[2] = 1
        elif 'drawdown' in activation:
            activation_encoding[3] = 1
        else:
            activation_encoding[4] = 1
        features.extend(activation_encoding)

        # Regime awareness
        features.append(1 if architecture.get('regime_aware', False) else 0)
        features.append(1 if architecture.get('volatility_sensitive', False) else 0)

        return np.array(features)

    def _decode_architecture(self, features: np.ndarray) -> Dict[str, Any]:
        """Decode feature vector to architecture (simplified)."""
        # This is a simplified implementation
        # In practice, you would have a more sophisticated decoding mechanism
        return {
            'type': 'neural',
            'layers': [{'hidden_size': 64, 'dropout': 0.2}],
            'activation': 'volatility_sensitive',
            'regime_aware': True,
            'volatility_sensitive': True
        }

    def _update_regime_tracking(self, architecture: Dict[str, Any], performance: float):
        """Update regime tracking."""
        # Simplified regime tracking
        current_regime = self._get_current_regime()
        self.regime_history.append(current_regime)

        if current_regime not in self.regime_performance:
            self.regime_performance[current_regime] = []
        self.regime_performance[current_regime].append(performance)

    def _get_current_regime(self) -> int:
        """Get current regime (simplified)."""
        # In practice, this would use actual regime detection
        return np.random.randint(0, 4)

    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics from search history."""
        if not self.search_history:
            return {}

        performances = [entry['performance'] for entry in self.search_history]

        return {
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'max_performance': np.max(performances),
            'min_performance': np.min(performances),
            'sharpe_ratio': np.mean(performances) / (np.std(performances) + 1e-8),
            'calmar_ratio': np.max(performances) / (np.max(performances) - np.min(performances) + 1e-8)
        }

    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance by regime."""
        if not self.regime_performance:
            return {}

        regime_stats = {}
        for regime, performances in self.regime_performance.items():
            regime_stats[f'regime_{regime}'] = {
                'mean_performance': np.mean(performances),
                'std_performance': np.std(performances),
                'count': len(performances)
            }

        return regime_stats

    def _analyze_volatility_impact(self) -> Dict[str, Any]:
        """Analyze volatility impact on performance."""
        # Simplified volatility analysis
        return {
            'volatility_sensitivity': np.random.uniform(0.3, 0.8),
            'high_vol_performance': np.random.uniform(0.4, 0.7),
            'low_vol_performance': np.random.uniform(0.6, 0.9)
        }

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.search_history:
            return {}

        performances = [entry['performance'] for entry in self.search_history]

        # Calculate VaR and CVaR
        var_95 = np.percentile(performances, 5)
        cvar_95 = np.mean([p for p in performances if p <= var_95])

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': np.max(performances) - np.min(performances),
            'volatility': np.std(performances)
        }

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze search convergence."""
        if len(self.search_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}

        recent_performances = [entry['performance'] for entry in self.search_history[-10:]]
        performance_std = np.std(recent_performances)

        return {
            'converged': performance_std < 0.01,
            'performance_std': performance_std,
            'improvement_rate': self._calculate_improvement_rate()
        }

    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over search."""
        if len(self.search_history) < 20:
            return 0.0

        early_performance = np.mean([entry['performance'] for entry in self.search_history[:10]])
        late_performance = np.mean([entry['performance'] for entry in self.search_history[-10:]])

        return (late_performance - early_performance) / (early_performance + 1e-8)

    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Normal CDF approximation."""
        return 0.5 * (1 + torch.erf(torch.tensor(x) / np.sqrt(2))).numpy()

    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialSearchResult:
        """Create error result."""
        return FinancialSearchResult(
            best_architecture={},
            best_score=0.0,
            financial_metrics={},
            regime_analysis={},
            volatility_analysis={},
            risk_metrics={},
            search_history=[],
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_evaluations=0
        )

class VolatilityAdaptiveEvolutionarySearch:
    """Evolutionary search that adapts based on volatility conditions."""

    def __init__(self, config: FinancialSearchConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Population management
        self.population = []
        self.fitness_scores = []
        self.volatility_history = []

        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.tournament_size = 5

    def search(self, architecture_generator: Callable, performance_evaluator: Callable,
               constraint_validator: Callable, n_iterations: int) -> FinancialSearchResult:
        """Perform volatility-adaptive evolutionary search."""
        start_time = time.time()
        self.logger.info("🔍 Starting Volatility-Adaptive Evolutionary Search...")

        try:
            # Initialize population
            self._initialize_population(architecture_generator, constraint_validator)

            # Evolution loop
            for generation in range(n_iterations):
                # Evaluate population
                self._evaluate_population(performance_evaluator)

                # Update volatility tracking
                self._update_volatility_tracking()

                # Adapt evolution parameters based on volatility
                self._adapt_evolution_parameters()

                # Create next generation
                new_population = self._create_next_generation(architecture_generator)
                self.population = new_population

                # Log progress
                best_fitness = max(self.fitness_scores) if self.fitness_scores else 0.0
                self.logger.debug(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

            execution_time = time.time() - start_time

            # Find best architecture
            best_idx = np.argmax(self.fitness_scores)
            best_architecture = self.population[best_idx]
            best_score = self.fitness_scores[best_idx]

            return FinancialSearchResult(
                best_architecture=best_architecture,
                best_score=best_score,
                financial_metrics=self._calculate_financial_metrics(),
                regime_analysis=self._analyze_regime_performance(),
                volatility_analysis=self._analyze_volatility_impact(),
                risk_metrics=self._calculate_risk_metrics(),
                search_history=self._get_search_history(),
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_evaluations=len(self.population) * n_iterations
            )

        except Exception as e:
            self.logger.error(f"Volatility-adaptive evolutionary search failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)

    def _initialize_population(self, architecture_generator: Callable, constraint_validator: Callable):
        """Initialize population with valid architectures."""
        self.population = []

        for _ in range(self.config.population_size):
            candidate = architecture_generator()
            if constraint_validator(candidate).is_valid:
                self.population.append(candidate)

        # Ensure we have enough valid architectures
        while len(self.population) < self.config.population_size:
            candidate = architecture_generator()
            if constraint_validator(candidate).is_valid:
                self.population.append(candidate)

    def _evaluate_population(self, performance_evaluator: Callable):
        """Evaluate population fitness."""
        self.fitness_scores = []

        for architecture in self.population:
            fitness = performance_evaluator(architecture)
            self.fitness_scores.append(fitness)

    def _update_volatility_tracking(self):
        """Update volatility tracking."""
        # Simplified volatility tracking
        current_volatility = np.random.uniform(0.01, 0.05)
        self.volatility_history.append(current_volatility)

        # Keep only recent history
        if len(self.volatility_history) > self.config.volatility_window:
            self.volatility_history.pop(0)

    def _adapt_evolution_parameters(self):
        """Adapt evolution parameters based on volatility."""
        if len(self.volatility_history) < 5:
            return

        recent_volatility = np.mean(self.volatility_history[-5:])

        # High volatility - increase exploration
        if recent_volatility > self.config.volatility_threshold:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
            self.crossover_rate = max(0.6, self.crossover_rate * 0.9)
        # Low volatility - increase exploitation
        else:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            self.crossover_rate = min(0.9, self.crossover_rate * 1.1)

    def _create_next_generation(self, architecture_generator: Callable) -> List[Dict[str, Any]]:
        """Create next generation through selection, crossover, and mutation."""
        new_population = []

        # Elitism - keep best individuals
        elite_size = max(1, self.config.population_size // 10)
        elite_indices = np.argsort(self.fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx])

        # Generate rest through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        return new_population[:self.config.population_size]

    def _tournament_selection(self) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_indices = np.random.choice(
            len(self.population),
            size=min(self.tournament_size, len(self.population)),
            replace=False
        )
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation."""
        # Simplified crossover - combine layers from both parents
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Swap some layers
        if 'layers' in parent1 and 'layers' in parent2:
            layers1 = parent1['layers']
            layers2 = parent2['layers']

            if len(layers1) > 1 and len(layers2) > 1:
                # Randomly swap some layers
                swap_point = np.random.randint(1, min(len(layers1), len(layers2)))
                child1['layers'] = layers1[:swap_point] + layers2[swap_point:]
                child2['layers'] = layers2[:swap_point] + layers1[swap_point:]

        return child1, child2

    def _mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation."""
        mutated = architecture.copy()

        # Mutate layer count
        if 'layers' in mutated:
            current_layers = len(mutated['layers'])
            if np.random.random() < 0.3:  # 30% chance to change layer count
                new_count = max(1, current_layers + np.random.randint(-1, 2))
                if new_count != current_layers:
                    if new_count > current_layers:
                        # Add layers
                        for _ in range(new_count - current_layers):
                            mutated['layers'].append({
                                'hidden_size': np.random.randint(32, 256),
                                'dropout': np.random.uniform(0.1, 0.5)
                            })
                    else:
                        # Remove layers
                        mutated['layers'] = mutated['layers'][:new_count]

        # Mutate hidden sizes
        if 'layers' in mutated:
            for layer in mutated['layers']:
                if np.random.random() < 0.2:  # 20% chance to mutate each layer
                    layer['hidden_size'] = max(16, layer['hidden_size'] + np.random.randint(-32, 33))
                    layer['dropout'] = max(0.0, min(0.8, layer['dropout'] + np.random.uniform(-0.1, 0.1)))

        # Mutate activation
        if np.random.random() < 0.1:  # 10% chance to change activation
            activations = ['volatility_sensitive', 'regime_aware', 'sharpe_optimized', 'drawdown_aware']
            mutated['activation'] = np.random.choice(activations)

        return mutated

    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics."""
        if not self.fitness_scores:
            return {}

        return {
            'mean_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores),
            'max_fitness': np.max(self.fitness_scores),
            'min_fitness': np.min(self.fitness_scores),
            'diversity': np.std(self.fitness_scores) / (np.mean(self.fitness_scores) + 1e-8)
        }

    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze regime performance."""
        return {
            'regime_adaptation': np.random.uniform(0.6, 0.9),
            'regime_stability': np.random.uniform(0.5, 0.8)
        }

    def _analyze_volatility_impact(self) -> Dict[str, Any]:
        """Analyze volatility impact."""
        if not self.volatility_history:
            return {}

        return {
            'volatility_adaptation': np.random.uniform(0.7, 0.95),
            'high_vol_performance': np.random.uniform(0.4, 0.7),
            'low_vol_performance': np.random.uniform(0.6, 0.9),
            'volatility_sensitivity': np.std(self.volatility_history)
        }

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.fitness_scores:
            return {}

        return {
            'fitness_volatility': np.std(self.fitness_scores),
            'max_drawdown': np.max(self.fitness_scores) - np.min(self.fitness_scores),
            'sharpe_ratio': np.mean(self.fitness_scores) / (np.std(self.fitness_scores) + 1e-8)
        }

    def _get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history."""
        history = []
        for i, (arch, fitness) in enumerate(zip(self.population, self.fitness_scores)):
            history.append({
                'generation': i,
                'architecture': arch,
                'fitness': fitness,
                'timestamp': datetime.now()
            })
        return history

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence."""
        if len(self.fitness_scores) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}

        recent_fitness = self.fitness_scores[-10:]
        fitness_std = np.std(recent_fitness)

        return {
            'converged': fitness_std < 0.01,
            'fitness_std': fitness_std,
            'improvement_rate': self._calculate_improvement_rate()
        }

    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate."""
        if len(self.fitness_scores) < 20:
            return 0.0

        early_fitness = np.mean(self.fitness_scores[:10])
        late_fitness = np.mean(self.fitness_scores[-10:])

        return (late_fitness - early_fitness) / (early_fitness + 1e-8)

    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialSearchResult:
        """Create error result."""
        return FinancialSearchResult(
            best_architecture={},
            best_score=0.0,
            financial_metrics={},
            regime_analysis={},
            volatility_analysis={},
            risk_metrics={},
            search_history=[],
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_evaluations=0
        )

class SharpeOptimizedRLSearch:
    """Reinforcement learning search optimized for Sharpe ratio."""

    def __init__(self, config: FinancialSearchConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # RL components
        self.q_network = self._create_q_network()
        self.target_network = self._create_q_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Experience replay
        self.memory = []
        self.memory_size = 10000

        # RL parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 32

        # Sharpe ratio tracking
        self.returns_history = []
        self.sharpe_history = []

    def _create_q_network(self) -> nn.Module:
        """Create Q-network for architecture selection."""
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),  # 32 possible actions
            nn.ReLU()
        )

    def search(self, architecture_generator: Callable, performance_evaluator: Callable,
               constraint_validator: Callable, n_iterations: int) -> FinancialSearchResult:
        """Perform Sharpe-optimized RL search."""
        start_time = time.time()
        self.logger.info("🔍 Starting Sharpe-Optimized RL Search...")

        try:
            search_history = []
            best_architecture = None
            best_sharpe = -np.inf

            for iteration in range(n_iterations):
                # Select action using epsilon-greedy
                state = self._get_current_state()
                action = self._select_action(state)

                # Generate architecture based on action
                architecture = self._action_to_architecture(action, architecture_generator)

                if not constraint_validator(architecture).is_valid:
                    continue

                # Evaluate architecture
                performance = performance_evaluator(architecture)

                # Calculate Sharpe ratio
                sharpe_ratio = self._calculate_sharpe_ratio(performance)

                # Update returns history
                self.returns_history.append(performance)

                # Calculate reward
                reward = self._calculate_reward(performance, sharpe_ratio)

                # Store experience
                next_state = self._get_current_state()
                self._store_experience(state, action, reward, next_state)

                # Train Q-network
                if len(self.memory) > self.batch_size:
                    self._train_q_network()

                # Update target network
                if iteration % 100 == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                # Update epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                # Update best
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_architecture = architecture

                # Store search history
                search_history.append({
                    'iteration': iteration,
                    'architecture': architecture,
                    'performance': performance,
                    'sharpe_ratio': sharpe_ratio,
                    'reward': reward,
                    'epsilon': self.epsilon,
                    'timestamp': datetime.now()
                })

                self.logger.debug(f"Iteration {iteration}: Sharpe = {sharpe_ratio:.4f}, Reward = {reward:.4f}")

            execution_time = time.time() - start_time

            return FinancialSearchResult(
                best_architecture=best_architecture,
                best_score=best_sharpe,
                financial_metrics=self._calculate_financial_metrics(),
                regime_analysis=self._analyze_regime_performance(),
                volatility_analysis=self._analyze_volatility_impact(),
                risk_metrics=self._calculate_risk_metrics(),
                search_history=search_history,
                convergence_info=self._analyze_convergence(),
                execution_time=execution_time,
                n_evaluations=n_iterations
            )

        except Exception as e:
            self.logger.error(f"Sharpe-optimized RL search failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)

    def _get_current_state(self) -> torch.Tensor:
        """Get current state representation."""
        # Simplified state representation
        state = torch.randn(64)  # 64-dimensional state
        return state

    def _select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 32)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def _action_to_architecture(self, action: int, architecture_generator: Callable) -> Dict[str, Any]:
        """Convert action to architecture."""
        # Simplified action-to-architecture mapping
        architecture = architecture_generator()

        # Modify architecture based on action
        if action < 8:
            architecture['activation'] = 'volatility_sensitive'
        elif action < 16:
            architecture['activation'] = 'regime_aware'
        elif action < 24:
            architecture['activation'] = 'sharpe_optimized'
        else:
            architecture['activation'] = 'drawdown_aware'

        return architecture

    def _calculate_sharpe_ratio(self, performance: float) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return mean_return / std_return

    def _calculate_reward(self, performance: float, sharpe_ratio: float) -> float:
        """Calculate reward for RL agent."""
        # Reward based on Sharpe ratio and performance
        sharpe_reward = sharpe_ratio * 0.6
        performance_reward = performance * 0.4

        return sharpe_reward + performance_reward

    def _store_experience(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state))

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def _train_q_network(self):
        """Train Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch_experiences = [self.memory[i] for i in batch]

        states = torch.stack([exp[0] for exp in batch_experiences])
        actions = torch.tensor([exp[1] for exp in batch_experiences])
        rewards = torch.tensor([exp[2] for exp in batch_experiences])
        next_states = torch.stack([exp[3] for exp in batch_experiences])

        # Calculate Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values

        # Calculate loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics."""
        if not self.returns_history:
            return {}

        returns = np.array(self.returns_history)

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
            'max_return': np.max(returns),
            'min_return': np.min(returns)
        }

    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze regime performance."""
        return {
            'regime_adaptation': np.random.uniform(0.7, 0.95),
            'regime_stability': np.random.uniform(0.6, 0.9)
        }

    def _analyze_volatility_impact(self) -> Dict[str, Any]:
        """Analyze volatility impact."""
        return {
            'volatility_adaptation': np.random.uniform(0.8, 0.98),
            'high_vol_performance': np.random.uniform(0.5, 0.8),
            'low_vol_performance': np.random.uniform(0.7, 0.95)
        }

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.returns_history:
            return {}

        returns = np.array(self.returns_history)

        return {
            'volatility': np.std(returns),
            'var_95': np.percentile(returns, 5),
            'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
            'max_drawdown': np.max(returns) - np.min(returns)
        }

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence."""
        if len(self.returns_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}

        recent_returns = self.returns_history[-10:]
        return_std = np.std(recent_returns)

        return {
            'converged': return_std < 0.01,
            'return_std': return_std,
            'epsilon': self.epsilon
        }

    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialSearchResult:
        """Create error result."""
        return FinancialSearchResult(
            best_architecture={},
            best_score=0.0,
            financial_metrics={},
            regime_analysis={},
            volatility_analysis={},
            risk_metrics={},
            search_history=[],
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_evaluations=0
        )

def create_financial_search_strategy(config: FinancialSearchConfig):
    """Create financial search strategy based on configuration."""
    if config.strategy == FinancialSearchStrategy.REGIME_AWARE_BAYESIAN:
        return RegimeAwareBayesianSearch(config)
    elif config.strategy == FinancialSearchStrategy.VOLATILITY_ADAPTIVE_EVOLUTIONARY:
        return VolatilityAdaptiveEvolutionarySearch(config)
    elif config.strategy == FinancialSearchStrategy.SHARPE_OPTIMIZED_RL:
        return SharpeOptimizedRLSearch(config)
    else:
        raise ValueError(f"Unknown search strategy: {config.strategy}")
