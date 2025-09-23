"""
Evolutionary Search Strategy

This module implements evolutionary algorithms for neural architecture search,
including genetic algorithms and evolutionary strategies.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy

from ..search.search_space import SearchSpace, ArchitectureConfig
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class EvolutionarySearchConfig:
    """Configuration for evolutionary search."""
    population_size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    max_generations: int = 100
    fitness_pressure: float = 1.0  # Higher = more selective

class EvolutionarySearch:
    """
    Evolutionary Search Strategy

    Implements evolutionary algorithms for neural architecture search,
    using genetic operators like selection, crossover, and mutation.
    """

    def __init__(self, config: EvolutionarySearchConfig):
        """Initialize evolutionary search.

        Args:
            config: Evolutionary search configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Initialize components
        self.search_space = SearchSpace()
        self.nas_utils = NASUtils()

        # Population state
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')

        # Statistics
        self.fitness_history = []
        self.diversity_history = []

        self.logger.info("🧬 Evolutionary search initialized")

    def generate_architecture(self, iteration: int) -> Optional[ArchitectureConfig]:
        """
        Generate architecture using evolutionary search.

        Args:
            iteration: Current search iteration

        Returns:
            Evolutionary architecture configuration or None
        """
        try:
            # Initialize population if empty
            if not self.population:
                self._initialize_population()
                return self.nas_utils.random_choice(self.population)

            # Generate new architecture through evolution
            return self._evolve_architecture()

        except Exception as e:
            self.logger.error(f"❌ Failed to generate architecture at iteration {iteration}: {e}")
            return None

    def _initialize_population(self):
        """Initialize population with random architectures."""
        self.logger.info("🌱 Initializing population")

        for i in range(self.config.population_size):
            input_dim = 100  # Default for market data
            output_dim = 5   # Default for regime detection

            architecture = self.search_space.generate_random_architecture(
                input_dim=input_dim,
                output_dim=output_dim,
                problem_type="regime_detection"
            )

            architecture.name = f"evo_gen{self.generation}_ind{i}"
            self.population.append(architecture)

        self.generation = 0
        self.logger.info(f"✅ Population initialized with {len(self.population)} individuals")

    def _evolve_architecture(self) -> ArchitectureConfig:
        """Generate new architecture through evolution.

        Returns:
            Evolved architecture configuration
        """
        if not self.population:
            return self._initialize_population()[0]

        # Select parents
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()

        # Crossover
        if self.nas_utils.random_float() < self.config.crossover_rate:
            offspring = self._crossover(parent1, parent2)
        else:
            offspring = deepcopy(parent1)

        # Mutation
        offspring = self._mutate(offspring)

        # Update generation
        offspring.name = f"evo_gen{self.generation}_offspring"
        offspring.calculate_complexity()
        offspring.estimate_parameters()

        self.logger.debug(f"🧬 Generated offspring: {offspring.name}")
        return offspring

    def _tournament_selection(self) -> ArchitectureConfig:
        """Select individual using tournament selection.

        Returns:
            Selected architecture configuration
        """
        # Randomly select tournament participants
        tournament_indices = np.random.choice(
            len(self.population),
            size=self.config.tournament_size,
            replace=False
        )

        # Find best in tournament
        best_individual = None
        best_fitness = float('-inf')

        for idx in tournament_indices:
            individual = self.population[idx]
            fitness = self.fitness_scores[idx] if self.fitness_scores else 0.0

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual

        return best_individual

    def _crossover(self, parent1: ArchitectureConfig, parent2: ArchitectureConfig) -> ArchitectureConfig:
        """Perform crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Offspring architecture configuration
        """
        offspring = ArchitectureConfig(
            name=f"{parent1.name}_x_{parent2.name}",
            input_dim=parent1.input_dim,
            output_dim=parent1.output_dim,
            hidden_dims=parent1.hidden_dims.copy(),
            activation=parent1.activation,
            dropout_rate=parent1.dropout_rate,
            batch_norm=parent1.batch_norm,
            use_residual=parent1.use_residual,
            problem_type=parent1.problem_type,
            layer_types=parent1.layer_types.copy(),
            attention_heads=parent1.attention_heads,
            embed_dim=parent1.embed_dim,
            use_attention=parent1.use_attention,
            use_lstm=parent1.use_lstm,
            use_convolution=parent1.use_convolution,
            num_layers=parent1.num_layers
        )

        # Crossover hidden dimensions
        if self.nas_utils.random_float() < 0.5:
            offspring.hidden_dims = parent2.hidden_dims.copy()

        # Crossover activation
        if self.nas_utils.random_float() < 0.3:
            offspring.activation = parent2.activation

        # Crossover dropout
        if self.nas_utils.random_float() < 0.3:
            offspring.dropout_rate = parent2.dropout_rate

        # Crossover boolean features
        if self.nas_utils.random_float() < 0.2:
            offspring.batch_norm = parent2.batch_norm

        if self.nas_utils.random_float() < 0.2:
            offspring.use_residual = parent2.use_residual

        if self.nas_utils.random_float() < 0.2:
            offspring.use_attention = parent2.use_attention

        if self.nas_utils.random_float() < 0.2:
            offspring.use_lstm = parent2.use_lstm

        if self.nas_utils.random_float() < 0.2:
            offspring.use_convolution = parent2.use_convolution

        # Crossover attention parameters
        if self.nas_utils.random_float() < 0.2:
            offspring.attention_heads = parent2.attention_heads

        if self.nas_utils.random_float() < 0.2:
            offspring.embed_dim = parent2.embed_dim

        return offspring

    def _mutate(self, individual: ArchitectureConfig) -> ArchitectureConfig:
        """Apply mutation to individual.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated architecture configuration
        """
        mutated = deepcopy(individual)

        # Mutate hidden dimensions
        if self.nas_utils.random_float() < self.config.mutation_rate:
            mutated.hidden_dims = self._mutate_hidden_dims(mutated.hidden_dims)

        # Mutate activation
        if self.nas_utils.random_float() < self.config.mutation_rate:
            mutated.activation = self.nas_utils.random_choice(self.search_space.activation_options)

        # Mutate dropout
        if self.nas_utils.random_float() < self.config.mutation_rate:
            dropout_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            mutated.dropout_rate = self.nas_utils.random_choice(dropout_options)

        # Mutate boolean features
        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated.batch_norm = not mutated.batch_norm

        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated.use_residual = not mutated.use_residual

        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated.use_attention = not mutated.use_attention

        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated.use_lstm = not mutated.use_lstm

        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated.use_convolution = not mutated.use_convolution

        # Mutate attention parameters
        if mutated.use_attention and self.nas_utils.random_float() < self.config.mutation_rate:
            heads_options = [1, 2, 4, 8, 16]
            mutated.attention_heads = self.nas_utils.random_choice(heads_options)

        if self.nas_utils.random_float() < self.config.mutation_rate:
            embed_options = [32, 64, 128, 256]
            mutated.embed_dim = self.nas_utils.random_choice(embed_options)

        return mutated

    def _mutate_hidden_dims(self, hidden_dims: List[int]) -> List[int]:
        """Mutate hidden layer dimensions.

        Args:
            hidden_dims: Current hidden dimensions

        Returns:
            Mutated hidden dimensions
        """
        mutated_dims = hidden_dims.copy()

        if not mutated_dims:
            return mutated_dims

        # Choose mutation type
        mutation_type = self.nas_utils.random_choice(['change', 'add', 'remove', 'swap'])

        if mutation_type == 'change' and mutated_dims:
            # Change one dimension
            idx = self.nas_utils.random_int(0, len(mutated_dims) - 1)
            dimension_options = [16, 32, 64, 128, 256, 512]
            new_dim = self.nas_utils.random_choice(dimension_options)
            mutated_dims[idx] = new_dim

        elif mutation_type == 'add' and len(mutated_dims) < 4:
            # Add a new layer
            dimension_options = [16, 32, 64, 128, 256]
            new_dim = self.nas_utils.random_choice(dimension_options)
            insert_idx = self.nas_utils.random_int(0, len(mutated_dims))
            mutated_dims.insert(insert_idx, new_dim)

        elif mutation_type == 'remove' and len(mutated_dims) > 1:
            # Remove a layer
            remove_idx = self.nas_utils.random_int(0, len(mutated_dims) - 1)
            mutated_dims.pop(remove_idx)

        elif mutation_type == 'swap' and len(mutated_dims) >= 2:
            # Swap two dimensions
            idx1, idx2 = np.random.choice(len(mutated_dims), 2, replace=False)
            mutated_dims[idx1], mutated_dims[idx2] = mutated_dims[idx2], mutated_dims[idx1]

        return mutated_dims

    def update_population(self, evaluated_architectures: List[Tuple[ArchitectureConfig, float]]):
        """Update population with evaluated architectures.

        Args:
            evaluated_architectures: List of (architecture, fitness) tuples
        """
        if not evaluated_architectures:
            return

        # Update fitness scores
        for architecture, fitness in evaluated_architectures:
            # Find architecture in population
            for i, pop_arch in enumerate(self.population):
                if pop_arch.name == architecture.name:
                    self.fitness_scores[i] = fitness
                    break

        # Update best individual
        best_idx = np.argmax(self.fitness_scores) if self.fitness_scores else 0
        if self.fitness_scores:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_individual = self.population[best_idx]

        # Create new generation
        self._create_new_generation()

        # Update statistics
        self._update_statistics()

    def _create_new_generation(self):
        """Create new generation using genetic operators."""
        new_population = []
        new_fitness = []

        # Elitism - keep best individuals
        elite_indices = np.argsort(self.fitness_scores)[-self.config.elite_size:]
        for idx in elite_indices:
            new_population.append(deepcopy(self.population[idx]))
            new_fitness.append(self.fitness_scores[idx])

        # Fill rest of population
        while len(new_population) < self.config.population_size:
            # Generate offspring
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            if self.nas_utils.random_float() < self.config.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = deepcopy(parent1)

            offspring = self._mutate(offspring)
            offspring.name = f"evo_gen{self.generation}_ind{len(new_population)}"

            new_population.append(offspring)
            new_fitness.append(0.0)  # Will be evaluated later

        # Update population
        self.population = new_population
        self.fitness_scores = new_fitness
        self.generation += 1

    def _update_statistics(self):
        """Update search statistics."""
        if self.fitness_scores:
            mean_fitness = np.mean(self.fitness_scores)
            std_fitness = np.std(self.fitness_scores)
            max_fitness = np.max(self.fitness_scores)
            min_fitness = np.min(self.fitness_scores)

            # Population diversity (simple measure)
            diversity = self._calculate_diversity()

            self.fitness_history.append({
                'generation': self.generation,
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'max_fitness': max_fitness,
                'min_fitness': min_fitness,
                'diversity': diversity
            })

            self.diversity_history.append(diversity)

            self.logger.debug(
                f"📈 Gen {self.generation}: "
                f"Mean: {mean_fitness:.4f}, "
                f"Max: {max_fitness:.4f}, "
                f"Diversity: {diversity:.4f}"
            )

    def _calculate_diversity(self) -> float:
        """Calculate population diversity.

        Returns:
            Diversity score
        """
        if len(self.population) < 2:
            return 0.0

        # Simple diversity based on architecture differences
        diversity_scores = []

        for i, arch1 in enumerate(self.population):
            for j, arch2 in enumerate(self.population):
                if i < j:
                    # Compare key architectural features
                    diff_score = 0.0

                    # Hidden dims difference
                    if arch1.hidden_dims != arch2.hidden_dims:
                        diff_score += 1.0

                    # Activation difference
                    if arch1.activation != arch2.activation:
                        diff_score += 0.5

                    # Dropout difference
                    if abs(arch1.dropout_rate - arch2.dropout_rate) > 0.1:
                        diff_score += 0.3

                    # Boolean features
                    bool_features = ['batch_norm', 'use_residual', 'use_attention', 'use_lstm', 'use_convolution']
                    for feature in bool_features:
                        if getattr(arch1, feature) != getattr(arch2, feature):
                            diff_score += 0.2

                    diversity_scores.append(diff_score)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary with search statistics
        """
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_individual_name': self.best_individual.name if self.best_individual else None,
            'mean_fitness': np.mean(self.fitness_scores) if self.fitness_scores else 0.0,
            'fitness_history_length': len(self.fitness_history),
            'diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'elite_size': self.config.elite_size,
            'mutation_rate': self.config.mutation_rate,
            'crossover_rate': self.config.crossover_rate
        }

    def reset_search(self):
        """Reset search state."""
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.diversity_history = []
        self.logger.info("🔄 Evolutionary search reset")