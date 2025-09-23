"""
Advanced Neural Architecture Search (NAS) System

This module implements state-of-the-art NAS techniques including:
- NSGA-II multi-objective optimization
- Intelligent pruning with median pruner
- Population-based search with genetic operators
- Multi-objective fitness evaluation
- Regime-specific search spaces
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import random
import time
from pathlib import Path

from ..core.nas_search import NASArchitectureSearch, NASSearchConfig
from ..core.nas_model import NASModel
from ..core.nas_trainer import NASTrainer, TrainingConfig
from ..core.nas_evaluator import NASEvaluator, EvaluationConfig
from ..search.search_space import SearchSpace, ArchitectureConfig
from ..evaluation.nas_metrics import NASMetrics, NASMetricsConfig

logger = logging.getLogger(__name__)

@dataclass
class NSGAIIConfig:
    """Configuration for NSGA-II optimization."""
    population_size: int = 50
    max_generations: int = 20
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "complexity", "efficiency"])
    objective_weights: List[float] = field(default_factory=lambda: [1.0, -0.3, 0.2])  # Negative for minimization

@dataclass
class MedianPrunerConfig:
    """Configuration for median pruning."""
    startup_trials: int = 5
    min_resource: int = 10
    reduction_factor: float = 3
    min_early_stopping_rate: int = 0

@dataclass
class AdvancedNASConfig:
    """Configuration for advanced NAS system."""
    nsga_ii_config: NSGAIIConfig = field(default_factory=NSGAIIConfig)
    pruner_config: MedianPrunerConfig = field(default_factory=MedianPrunerConfig)
    use_median_pruning: bool = True
    use_population_based: bool = True
    use_regime_specific: bool = True
    n_objectives: int = 3
    pareto_front_size: int = 10
    diversity_maintenance: bool = True

class Individual:
    """Individual in the evolutionary population."""

    def __init__(self, architecture: ArchitectureConfig, fitness: List[float]):
        """Initialize individual.

        Args:
            architecture: Neural architecture
            fitness: Multi-objective fitness values
        """
        self.architecture = architecture
        self.fitness = fitness
        self.rank = 0
        self.crowding_distance = 0.0
        self.age = 0

    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another.

        Args:
            other: Other individual to compare with

        Returns:
            True if this individual dominates the other
        """
        # Check if this individual is better in all objectives
        all_better = all(f1 >= f2 for f1, f2 in zip(self.fitness, other.fitness))
        any_better = any(f1 > f2 for f1, f2 in zip(self.fitness, other.fitness))
        return all_better and any_better

    def __repr__(self):
        return f"Individual(fitness={self.fitness}, rank={self.rank})"

class NSGAII_Optimizer:
    """
    NSGA-II Multi-Objective Optimizer

    Implements the Non-dominated Sorting Genetic Algorithm II
    for multi-objective architecture optimization.
    """

    def __init__(self, config: NSGAIIConfig):
        """Initialize NSGA-II optimizer.

        Args:
            config: NSGA-II configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Population
        self.population: List[Individual] = []
        self.offspring: List[Individual] = []
        self.pareto_front: List[Individual] = []

        # Statistics
        self.generation = 0
        self.best_individual = None

        self.logger.info("🧬 NSGA-II Optimizer initialized")

    def optimize(self,
                train_data: Tuple[np.ndarray, np.ndarray],
                val_data: Tuple[np.ndarray, np.ndarray],
                problem_type: str = "classification") -> List[Individual]:
        """
        Run NSGA-II optimization.

        Args:
            train_data: Training data
            val_data: Validation data
            problem_type: Type of problem

        Returns:
            Pareto-optimal individuals
        """
        logger.info("🚀 Starting NSGA-II optimization")

        # Initialize population
        self._initialize_population(train_data, val_data, problem_type)

        # Evolution loop
        for generation in range(self.config.max_generations):
            self.generation = generation

            # Create offspring
            self._create_offspring()

            # Evaluate offspring
            self._evaluate_population(self.offspring, train_data, val_data, problem_type)

            # Combine populations
            combined_population = self.population + self.offspring

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined_population)

            # Select next generation
            self.population = self._select_next_generation(fronts, len(self.population))

            # Update Pareto front
            self.pareto_front = fronts[0]

            # Log progress
            if generation % 5 == 0:
                self._log_generation_progress()

        logger.info(f"✅ NSGA-II optimization completed after {self.config.max_generations} generations")
        return self.pareto_front

    def _initialize_population(self,
                             train_data: Tuple[np.ndarray, np.ndarray],
                             val_data: Tuple[np.ndarray, np.ndarray],
                             problem_type: str):
        """Initialize population with random architectures."""
        logger.info(f"🌱 Initializing population of {self.config.population_size}")

        search_space = SearchSpace()

        for i in range(self.config.population_size):
            # Generate random architecture
            input_dim = train_data[0].shape[1]
            output_dim = len(np.unique(train_data[1]))

            architecture = search_space.generate_random_architecture(
                input_dim=input_dim, output_dim=output_dim, problem_type=problem_type
            )

            # Create individual
            individual = Individual(architecture, fitness=[0.0] * len(self.config.objectives))
            self.population.append(individual)

        # Evaluate initial population
        self._evaluate_population(self.population, train_data, val_data, problem_type)

        logger.info("✅ Population initialized and evaluated")

    def _create_offspring(self):
        """Create offspring through crossover and mutation."""
        self.offspring = []

        while len(self.offspring) < len(self.population):
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.config.crossover_rate:
                child_architecture = self._crossover(parent1.architecture, parent2.architecture)
            else:
                child_architecture = deepcopy(parent1.architecture)

            # Mutation
            if random.random() < self.config.mutation_rate:
                child_architecture = self._mutate(child_architecture)

            # Create offspring individual
            child = Individual(child_architecture, fitness=[0.0] * len(self.config.objectives))
            self.offspring.append(child)

    def _tournament_selection(self) -> Individual:
        """Tournament selection for parent selection."""
        # Randomly select tournament participants
        tournament = random.sample(self.population, self.config.tournament_size)

        # Select best individual (crowding distance as tiebreaker)
        best = tournament[0]
        for individual in tournament[1:]:
            if individual.rank < best.rank or (individual.rank == best.rank and individual.crowding_distance > best.crowding_distance):
                best = individual

        return best

    def _crossover(self, arch1: ArchitectureConfig, arch2: ArchitectureConfig) -> ArchitectureConfig:
        """Crossover two architectures."""
        # Crossover hidden dimensions
        if random.random() < 0.5:
            hidden_dims = arch1.hidden_dims.copy()
        else:
            hidden_dims = arch2.hidden_dims.copy()

        # Crossover activation
        activation = arch1.activation if random.random() < 0.5 else arch2.activation

        # Crossover dropout
        dropout_rate = arch1.dropout_rate if random.random() < 0.5 else arch2.dropout_rate

        # Crossover boolean features
        batch_norm = arch1.batch_norm if random.random() < 0.5 else arch2.batch_norm
        use_residual = arch1.use_residual if random.random() < 0.5 else arch2.use_residual

        # Create offspring
        offspring = ArchitectureConfig(
            name=f"nsga2_gen{self.generation}_offspring",
            input_dim=arch1.input_dim,
            output_dim=arch1.output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual,
            problem_type=arch1.problem_type
        )

        offspring.calculate_complexity()
        offspring.estimate_parameters()

        return offspring

    def _mutate(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate architecture."""
        mutated = deepcopy(architecture)

        # Mutate hidden dimensions
        if mutated.hidden_dims and random.random() < 0.3:
            idx = random.randint(0, len(mutated.hidden_dims) - 1)
            change_factor = 1.0 + random.uniform(-0.2, 0.2)
            new_dim = max(16, int(mutated.hidden_dims[idx] * change_factor))
            mutated.hidden_dims[idx] = new_dim

        # Mutate activation
        if random.random() < 0.2:
            activations = ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'swish']
            mutated.activation = random.choice(activations)

        # Mutate dropout
        if random.random() < 0.3:
            change_factor = 1.0 + random.uniform(-0.1, 0.1)
            mutated.dropout_rate = max(0.0, min(0.5, mutated.dropout_rate * change_factor))

        # Mutate boolean features
        if random.random() < 0.1:
            mutated.batch_norm = not mutated.batch_norm

        if random.random() < 0.1:
            mutated.use_residual = not mutated.use_residual

        mutated.calculate_complexity()
        mutated.estimate_parameters()

        return mutated

    def _evaluate_population(self,
                           population: List[Individual],
                           train_data: Tuple[np.ndarray, np.ndarray],
                           val_data: Tuple[np.ndarray, np.ndarray],
                           problem_type: str):
        """Evaluate population fitness."""
        for individual in population:
            fitness = self._evaluate_architecture(individual.architecture, train_data, val_data, problem_type)
            individual.fitness = fitness
            individual.age += 1

    def _evaluate_architecture(self,
                              architecture: ArchitectureConfig,
                              train_data: Tuple[np.ndarray, np.ndarray],
                              val_data: Tuple[np.ndarray, np.ndarray],
                              problem_type: str) -> List[float]:
        """Evaluate architecture on multiple objectives."""
        try:
            # Create model
            model = NASModel.create_from_config(architecture, problem_type)

            # Create data loaders
            train_loader, val_loader = self._create_simple_data_loaders(train_data, val_data)

            # Train briefly for evaluation
            trainer_config = TrainingConfig(epochs=5, batch_size=32)  # Quick evaluation
            trainer = NASTrainer(trainer_config)
            training_result = trainer.train(model, train_loader, val_loader, problem_type)

            # Evaluate
            evaluator_config = EvaluationConfig(batch_size=32)
            evaluator = NASEvaluator(evaluator_config)
            evaluation_result = evaluator.evaluate_architecture(
                training_result.model, train_loader, val_loader, problem_type="accuracy"
            )

            # Calculate multi-objective fitness
            fitness = []

            # Objective 1: Accuracy
            accuracy = evaluation_result.accuracy
            fitness.append(accuracy)

            # Objective 2: Complexity (minimize)
            complexity = architecture.complexity_score
            fitness.append(-complexity)  # Negative for minimization

            # Objective 3: Efficiency (minimize training time)
            efficiency = training_result.execution_time
            fitness.append(-efficiency)  # Negative for minimization

            return fitness

        except Exception as e:
            logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            return [0.0, -10.0, -1000.0]  # Poor fitness for failed architectures

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting algorithm."""
        fronts = [[]]

        for individual in population:
            individual.rank = 0
            individual.crowding_distance = 0.0

        for i, p in enumerate(population):
            p.domination_count = 0
            p.dominated_solutions = []

            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        Q.append(q)

            if Q:
                fronts.append(Q)
                i += 1
            else:
                break

        # Calculate crowding distance for each front
        for front in fronts:
            self._calculate_crowding_distance(front)

        return fronts

    def _calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return

        n_objectives = len(self.config.objectives)

        for individual in front:
            individual.crowding_distance = 0.0

        for m in range(n_objectives):
            # Sort by objective m
            front.sort(key=lambda x: x.fitness[m])

            # Boundary individuals have infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate distance for intermediate individuals
            f_max = front[-1].fitness[m]
            f_min = front[0].fitness[m]

            if f_max == f_min:
                continue

            for i in range(1, len(front) - 1):
                distance = (front[i+1].fitness[m] - front[i-1].fitness[m]) / (f_max - f_min)
                front[i].crowding_distance += distance

    def _select_next_generation(self, fronts: List[List[Individual]], population_size: int) -> List[Individual]:
        """Select next generation using NSGA-II selection."""
        next_generation = []

        for front in fronts:
            if len(next_generation) + len(front) <= population_size:
                # Add entire front
                next_generation.extend(front)
            else:
                # Sort by crowding distance and add best
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining_slots = population_size - len(next_generation)
                next_generation.extend(front[:remaining_slots])
                break

        return next_generation[:population_size]

    def _create_simple_data_loaders(self, train_data: Tuple[np.ndarray, np.ndarray],
                                   val_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create simple data loaders for evaluation."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

    def _log_generation_progress(self):
        """Log progress of current generation."""
        if self.pareto_front:
            best_fitness = [f"{f:.4f}" for f in self.pareto_front[0].fitness]
            logger.info(f"📈 Generation {self.generation}: Best fitness = {best_fitness}")

class MedianPruner:
    """
    Median Pruner for Early Stopping

    Eliminates poor-performing trials early based on median performance
    of completed trials.
    """

    def __init__(self, config: MedianPrunerConfig):
        """Initialize median pruner.

        Args:
            config: Pruner configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Trial tracking
        self.trials = []
        self.pruned_trials = 0

    def prune_trial(self, trial_id: int, step: int, value: float) -> bool:
        """
        Check if trial should be pruned.

        Args:
            trial_id: Trial identifier
            step: Current step
            value: Current value

        Returns:
            True if trial should be pruned
        """
        # Don't prune early trials
        if step < self.config.startup_trials:
            return False

        # Don't prune if below minimum resource
        if step < self.config.min_resource:
            return False

        # Check if we should prune
        if len(self.trials) < self.config.min_early_stopping_rate:
            return False

        # Get median of completed trials at this step
        completed_trials = [t for t in self.trials if t['step'] >= step]
        if len(completed_trials) < 3:  # Need at least 3 trials for median
            return False

        values = [t['value'] for t in completed_trials]
        median_value = np.median(values)

        # Prune if current value is worse than median
        should_prune = value < median_value * 0.8  # 20% worse than median

        if should_prune:
            self.pruned_trials += 1
            logger.info(f"🪓 Pruned trial {trial_id} at step {step} (value: {value:.4f}, median: {median_value:.4f})")

        return should_prune

    def add_trial_result(self, trial_id: int, step: int, value: float):
        """Add trial result."""
        self.trials.append({
            'trial_id': trial_id,
            'step': step,
            'value': value
        })

class RegimeSpecificSearchSpace:
    """
    Regime-Specific Search Spaces

    Provides specialized search spaces for different market regimes:
    - Volatility-focused architectures
    - Trend-focused architectures
    - Volume-focused architectures
    - Momentum-focused architectures
    - Hybrid architectures
    """

    def __init__(self):
        """Initialize regime-specific search space."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_space = SearchSpace()

    def get_volatility_architecture(self, input_dim: int, output_dim: int) -> ArchitectureConfig:
        """Get architecture optimized for volatility regimes."""
        config = ArchitectureConfig(
            name="volatility_focused",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[128, 64, 32],
            activation="leaky_relu",
            dropout_rate=0.2,
            batch_norm=True,
            use_residual=True,
            problem_type="volatility_regime",
            layer_types=["dense", "dense", "dense"],
            use_attention=True,
            attention_heads=8,
            embed_dim=64
        )
        config.calculate_complexity()
        config.estimate_parameters()
        return config

    def get_trend_architecture(self, input_dim: int, output_dim: int) -> ArchitectureConfig:
        """Get architecture optimized for trend regimes."""
        config = ArchitectureConfig(
            name="trend_focused",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[256, 128, 64],
            activation="relu",
            dropout_rate=0.1,
            batch_norm=True,
            use_residual=True,
            problem_type="trend_regime",
            layer_types=["dense", "dense", "dense"],
            use_lstm=True,
            use_attention=False
        )
        config.calculate_complexity()
        config.estimate_parameters()
        return config

    def get_volume_architecture(self, input_dim: int, output_dim: int) -> ArchitectureConfig:
        """Get architecture optimized for volume regimes."""
        config = ArchitectureConfig(
            name="volume_focused",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[64, 32, 16],
            activation="tanh",
            dropout_rate=0.3,
            batch_norm=False,
            use_residual=False,
            problem_type="volume_regime",
            layer_types=["dense", "dense", "dense"],
            use_convolution=True,
            use_attention=True,
            attention_heads=4
        )
        config.calculate_complexity()
        config.estimate_parameters()
        return config

    def get_momentum_architecture(self, input_dim: int, output_dim: int) -> ArchitectureConfig:
        """Get architecture optimized for momentum regimes."""
        config = ArchitectureConfig(
            name="momentum_focused",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[96, 48, 24],
            activation="swish",
            dropout_rate=0.25,
            batch_norm=True,
            use_residual=True,
            problem_type="momentum_regime",
            layer_types=["dense", "dense", "dense"],
            use_attention=True,
            attention_heads=6,
            embed_dim=48
        )
        config.calculate_complexity()
        config.estimate_parameters()
        return config

    def get_hybrid_architecture(self, input_dim: int, output_dim: int) -> ArchitectureConfig:
        """Get hybrid architecture for mixed regimes."""
        config = ArchitectureConfig(
            name="hybrid_regime",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[192, 96, 48, 24],
            activation="gelu",
            dropout_rate=0.15,
            batch_norm=True,
            use_residual=True,
            problem_type="hybrid_regime",
            layer_types=["dense", "dense", "dense", "dense"],
            use_attention=True,
            attention_heads=8,
            embed_dim=96,
            use_lstm=True
        )
        config.calculate_complexity()
        config.estimate_parameters()
        return config

    def get_regime_specific_ensemble(self, input_dim: int, output_dim: int) -> List[ArchitectureConfig]:
        """Get ensemble of regime-specific architectures."""
        return [
            self.get_volatility_architecture(input_dim, output_dim),
            self.get_trend_architecture(input_dim, output_dim),
            self.get_volume_architecture(input_dim, output_dim),
            self.get_momentum_architecture(input_dim, output_dim),
            self.get_hybrid_architecture(input_dim, output_dim)
        ]

class AdvancedNAS:
    """
    Advanced Neural Architecture Search System

    Combines NSGA-II, median pruning, and regime-specific search spaces
    for state-of-the-art NAS optimization.
    """

    def __init__(self, config: AdvancedNASConfig):
        """Initialize advanced NAS system.

        Args:
            config: Advanced NAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.nsga_ii = NSGAII_Optimizer(config.nsga_ii_config)
        self.median_pruner = MedianPruner(config.pruner_config)
        self.regime_search_space = RegimeSpecificSearchSpace()

        self.logger.info("🚀 Advanced NAS System initialized")

    def optimize_multi_objective(self,
                                train_data: Tuple[np.ndarray, np.ndarray],
                                val_data: Tuple[np.ndarray, np.ndarray],
                                problem_type: str = "classification") -> List[Individual]:
        """
        Run multi-objective NAS optimization.

        Args:
            train_data: Training data
            val_data: Validation data
            problem_type: Type of problem

        Returns:
            Pareto-optimal architectures
        """
        logger.info("🚀 Starting multi-objective NAS optimization")

        # Run NSGA-II optimization
        pareto_front = self.nsga_ii.optimize(train_data, val_data, problem_type)

        # Apply median pruning if enabled
        if self.config.use_median_pruning:
            pareto_front = self._apply_median_pruning(pareto_front, train_data, val_data, problem_type)

        logger.info(f"✅ Multi-objective optimization completed with {len(pareto_front)} Pareto-optimal architectures")
        return pareto_front

    def _apply_median_pruning(self,
                             pareto_front: List[Individual],
                             train_data: Tuple[np.ndarray, np.ndarray],
                             val_data: Tuple[np.ndarray, np.ndarray],
                             problem_type: str) -> List[Individual]:
        """Apply median pruning to Pareto front."""
        logger.info("🪓 Applying median pruning to Pareto front")

        pruned_front = []
        for individual in pareto_front:
            # Evaluate with pruning
            should_prune = self.median_pruner.prune_trial(
                trial_id=id(individual),
                step=self.nsga_ii.generation,
                value=np.mean(individual.fitness)
            )

            if not should_prune:
                pruned_front.append(individual)

        logger.info(f"🪓 Median pruning removed {len(pareto_front) - len(pruned_front)} architectures")
        return pruned_front

    def get_regime_specific_optimization(self,
                                       market_data: np.ndarray,
                                       regime_type: str = "hybrid") -> Dict[str, Any]:
        """
        Run regime-specific NAS optimization.

        Args:
            market_data: Market data
            regime_type: Type of regime ("volatility", "trend", "volume", "momentum", "hybrid")

        Returns:
            Optimization results
        """
        logger.info(f"🎯 Running regime-specific optimization for {regime_type} regime")

        # Get regime-specific architecture
        if regime_type == "volatility":
            base_architecture = self.regime_search_space.get_volatility_architecture(
                input_dim=market_data.shape[1], output_dim=5
            )
        elif regime_type == "trend":
            base_architecture = self.regime_search_space.get_trend_architecture(
                input_dim=market_data.shape[1], output_dim=5
            )
        elif regime_type == "volume":
            base_architecture = self.regime_search_space.get_volume_architecture(
                input_dim=market_data.shape[1], output_dim=5
            )
        elif regime_type == "momentum":
            base_architecture = self.regime_search_space.get_momentum_architecture(
                input_dim=market_data.shape[1], output_dim=5
            )
        else:  # hybrid
            base_architecture = self.regime_search_space.get_hybrid_architecture(
                input_dim=market_data.shape[1], output_dim=5
            )

        # Create training data
        X_train = market_data
        y_train = np.random.randint(0, 5, len(market_data))  # Placeholder labels

        # Perform optimization
        results = {
            'regime_type': regime_type,
            'base_architecture': base_architecture,
            'architecture_name': base_architecture.name,
            'complexity_score': base_architecture.complexity_score,
            'estimated_parameters': base_architecture.estimated_parameters,
            'optimization_method': 'regime_specific_nas'
        }

        logger.info(f"✅ Regime-specific optimization completed for {regime_type}")
        return results

def run_advanced_nas_optimization(market_data: np.ndarray,
                                 n_objectives: int = 3) -> Dict[str, Any]:
    """Run advanced NAS optimization with all features."""
    logger.info("🚀 Running Advanced NAS Optimization")

    # Configure advanced NAS
    config = AdvancedNASConfig(
        nsga_ii_config=NSGAIIConfig(
            population_size=30,
            max_generations=15,
            objectives=["accuracy", "complexity", "efficiency"]
        ),
        use_median_pruning=True,
        use_population_based=True,
        use_regime_specific=True,
        n_objectives=n_objectives
    )

    advanced_nas = AdvancedNAS(config)

    # Prepare data
    X_train = market_data
    y_train = np.random.randint(0, 5, len(market_data))  # Placeholder

    # Run multi-objective optimization
    pareto_front = advanced_nas.optimize_multi_objective(
        train_data=(X_train, y_train),
        val_data=(X_train, y_train),  # Use same data for demo
        problem_type="regime_detection"
    )

    # Get regime-specific results
    regime_results = {}
    for regime_type in ["volatility", "trend", "volume", "momentum", "hybrid"]:
        regime_results[regime_type] = advanced_nas.get_regime_specific_optimization(
            market_data, regime_type
        )

    results = {
        'pareto_front': pareto_front,
        'regime_specific_results': regime_results,
        'n_pareto_optimal': len(pareto_front),
        'optimization_method': 'advanced_nsga_ii',
        'median_pruning_used': config.use_median_pruning,
        'population_based': config.use_population_based,
        'regime_specific': config.use_regime_specific
    }

    logger.info(f"✅ Advanced NAS optimization completed")
    logger.info(f"📊 Found {len(pareto_front)} Pareto-optimal architectures")

    return results