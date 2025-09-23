"""
Random Search Strategy

This module implements random search for neural architecture search,
which randomly samples architectures from the search space.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..search.search_space import SearchSpace, ArchitectureConfig
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class RandomSearchConfig:
    """Configuration for random search."""
    max_samples: int = 100
    sample_with_replacement: bool = True
    adaptive_sampling: bool = False
    mutation_rate: float = 0.1
    elite_fraction: float = 0.1

class RandomSearch:
    """
    Random Search Strategy

    Implements random search for neural architecture search by randomly
    sampling architectures from the defined search space.
    """

    def __init__(self, config: RandomSearchConfig):
        """Initialize random search.

        Args:
            config: Random search configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Initialize components
        self.search_space = SearchSpace()
        self.nas_utils = NASUtils()

        # Search state
        self.searched_architectures = []
        self.elite_architectures = []
        self.best_score = float('-inf')
        self.best_architecture = None

        # Adaptive sampling state
        self.sample_history = []
        self.mutation_history = []

        self.logger.info("🎲 Random search initialized")

    def generate_architecture(self, iteration: int) -> Optional[ArchitectureConfig]:
        """
        Generate a random architecture.

        Args:
            iteration: Current search iteration

        Returns:
            Random architecture configuration or None
        """
        try:
            # Use adaptive sampling if enabled
            if self.config.adaptive_sampling and iteration > 10:
                return self._generate_adaptive_architecture(iteration)
            else:
                return self._generate_random_architecture()

        except Exception as e:
            self.logger.error(f"❌ Failed to generate architecture at iteration {iteration}: {e}")
            return None

    def _generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a completely random architecture.

        Returns:
            Random architecture configuration
        """
        # Generate random parameters
        input_dim = self.nas_utils.random_choice(self.search_space.input_dims)
        output_dim = self.nas_utils.random_choice(self.search_space.output_dims)

        # For now, use fixed dimensions - in practice these would come from data
        input_dim = 100  # Default for market data
        output_dim = 5   # Default for regime detection

        # Generate architecture
        architecture = self.search_space.generate_random_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            problem_type="regime_detection"
        )

        self.logger.debug(f"🎲 Generated random architecture: {architecture.name}")
        return architecture

    def _generate_adaptive_architecture(self, iteration: int) -> ArchitectureConfig:
        """Generate architecture using adaptive sampling.

        Args:
            iteration: Current search iteration

        Returns:
            Adaptive architecture configuration
        """
        # Choose sampling strategy
        strategy = self._select_adaptive_strategy(iteration)

        if strategy == "mutation":
            return self._mutate_architecture()
        elif strategy == "elite_crossover":
            return self._crossover_elite_architectures()
        elif strategy == "local_search":
            return self._local_search_architecture()
        else:
            # Fallback to random
            return self._generate_random_architecture()

    def _select_adaptive_strategy(self, iteration: int) -> str:
        """Select adaptive sampling strategy based on iteration.

        Args:
            iteration: Current search iteration

        Returns:
            Strategy name
        """
        # Simple strategy selection - can be made more sophisticated
        if iteration % 10 == 0:  # Every 10 iterations
            return "mutation"
        elif len(self.elite_architectures) >= 3:
            return "elite_crossover"
        else:
            return "random"

    def _mutate_architecture(self) -> ArchitectureConfig:
        """Mutate an existing architecture.

        Returns:
            Mutated architecture configuration
        """
        if not self.elite_architectures:
            return self._generate_random_architecture()

        # Select parent architecture
        parent = self.nas_utils.random_choice(self.elite_architectures)

        # Create mutation
        mutated_config = ArchitectureConfig(
            name=f"{parent.name}_mut_{len(self.mutation_history)}",
            input_dim=parent.input_dim,
            output_dim=parent.output_dim,
            hidden_dims=parent.hidden_dims.copy(),
            activation=parent.activation,
            dropout_rate=parent.dropout_rate,
            batch_norm=parent.batch_norm,
            use_residual=parent.use_residual,
            problem_type=parent.problem_type,
            layer_types=parent.layer_types.copy(),
            attention_heads=parent.attention_heads,
            embed_dim=parent.embed_dim,
            use_attention=parent.use_attention,
            use_lstm=parent.use_lstm,
            use_convolution=parent.use_convolution,
            num_layers=parent.num_layers
        )

        # Apply mutations
        mutations_applied = 0

        # Mutate hidden dimensions
        if self.nas_utils.random_float() < self.config.mutation_rate:
            mutated_config.hidden_dims = self._mutate_hidden_dims(mutated_config.hidden_dims)
            mutations_applied += 1

        # Mutate activation
        if self.nas_utils.random_float() < self.config.mutation_rate:
            mutated_config.activation = self.nas_utils.random_choice(self.search_space.activation_options)
            mutations_applied += 1

        # Mutate dropout rate
        if self.nas_utils.random_float() < self.config.mutation_rate:
            mutated_config.dropout_rate = self.nas_utils.random_choice(self.search_space.dropout_options)
            mutations_applied += 1

        # Toggle batch norm
        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated_config.batch_norm = not mutated_config.batch_norm
            mutations_applied += 1

        # Toggle residual connections
        if self.nas_utils.random_float() < self.config.mutation_rate / 2:
            mutated_config.use_residual = not mutated_config.use_residual
            mutations_applied += 1

        # Recalculate complexity and parameters
        mutated_config.calculate_complexity()
        mutated_config.estimate_parameters()

        self.logger.debug(f"🧬 Applied {mutations_applied} mutations to {parent.name}")
        self.mutation_history.append(mutated_config.name)

        return mutated_config

    def _mutate_hidden_dims(self, hidden_dims: List[int]) -> List[int]:
        """Mutate hidden layer dimensions.

        Args:
            hidden_dims: Current hidden dimensions

        Returns:
            Mutated hidden dimensions
        """
        mutated_dims = hidden_dims.copy()

        # Randomly change one dimension
        if mutated_dims:
            idx = self.nas_utils.random_int(0, len(mutated_dims) - 1)
            current_dim = mutated_dims[idx]

            # Possible dimension changes
            dimension_options = [16, 32, 64, 128, 256, 512]
            new_dim = self.nas_utils.random_choice(dimension_options)

            mutated_dims[idx] = new_dim

        return mutated_dims

    def _crossover_elite_architectures(self) -> ArchitectureConfig:
        """Create new architecture by crossing over elite architectures.

        Returns:
            Crossover architecture configuration
        """
        if len(self.elite_architectures) < 2:
            return self.nas_utils.random_choice(self.elite_architectures)

        # Select two parents
        parent1 = self.nas_utils.random_choice(self.elite_architectures)
        parent2 = self.nas_utils.random_choice(self.elite_architectures)

        # Crossover parameters
        crossover_config = ArchitectureConfig(
            name=f"{parent1.name}_x_{parent2.name}_{len(self.sample_history)}",
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
            # Take from parent2
            crossover_config.hidden_dims = parent2.hidden_dims.copy()

        # Crossover activation
        if self.nas_utils.random_float() < 0.3:
            crossover_config.activation = parent2.activation

        # Crossover dropout
        if self.nas_utils.random_float() < 0.3:
            crossover_config.dropout_rate = parent2.dropout_rate

        # Crossover batch norm
        if self.nas_utils.random_float() < 0.2:
            crossover_config.batch_norm = parent2.batch_norm

        # Crossover residual
        if self.nas_utils.random_float() < 0.2:
            crossover_config.use_residual = parent2.use_residual

        # Recalculate metrics
        crossover_config.calculate_complexity()
        crossover_config.estimate_parameters()

        self.logger.debug(f"🔄 Created crossover architecture from {parent1.name} and {parent2.name}")
        self.sample_history.append(crossover_config.name)

        return crossover_config

    def _local_search_architecture(self) -> ArchitectureConfig:
        """Generate architecture using local search around best.

        Returns:
            Local search architecture configuration
        """
        if not self.best_architecture:
            return self._generate_random_architecture()

        # Start with best architecture and make small changes
        local_config = ArchitectureConfig(
            name=f"{self.best_architecture.name}_local_{len(self.sample_history)}",
            input_dim=self.best_architecture.input_dim,
            output_dim=self.best_architecture.output_dim,
            hidden_dims=self.best_architecture.hidden_dims.copy(),
            activation=self.best_architecture.activation,
            dropout_rate=self.best_architecture.dropout_rate,
            batch_norm=self.best_architecture.batch_norm,
            use_residual=self.best_architecture.use_residual,
            problem_type=self.best_architecture.problem_type,
            layer_types=self.best_architecture.layer_types.copy(),
            attention_heads=self.best_architecture.attention_heads,
            embed_dim=self.best_architecture.embed_dim,
            use_attention=self.best_architecture.use_attention,
            use_lstm=self.best_architecture.use_lstm,
            use_convolution=self.best_architecture.use_convolution,
            num_layers=self.best_architecture.num_layers
        )

        # Make small local changes
        if self.nas_utils.random_float() < 0.5 and local_config.hidden_dims:
            # Slightly modify hidden dimensions
            idx = self.nas_utils.random_int(0, len(local_config.hidden_dims) - 1)
            current_dim = local_config.hidden_dims[idx]

            # Small change (±25%)
            change_factor = 1.0 + self.nas_utils.random_float(-0.25, 0.25)
            new_dim = max(8, int(current_dim * change_factor))
            new_dim = min(new_dim, 1024)  # Keep within bounds

            local_config.hidden_dims[idx] = new_dim

        # Small change to dropout
        if self.nas_utils.random_float() < 0.3:
            change_factor = 1.0 + self.nas_utils.random_float(-0.2, 0.2)
            new_dropout = max(0.0, min(0.5, local_config.dropout_rate * change_factor))
            local_config.dropout_rate = round(new_dropout, 2)

        # Recalculate metrics
        local_config.calculate_complexity()
        local_config.estimate_parameters()

        self.logger.debug(f"🎯 Generated local search architecture around {self.best_architecture.name}")
        self.sample_history.append(local_config.name)

        return local_config

    def update_best(self, architecture: ArchitectureConfig, score: float):
        """Update the best architecture found.

        Args:
            architecture: Architecture configuration
            score: Architecture score
        """
        if score > self.best_score:
            self.best_score = score
            self.best_architecture = architecture
            self.logger.info(f"🎯 New best architecture: {architecture.name} (score: {score:.4f})")

    def add_to_elite(self, architecture: ArchitectureConfig):
        """Add architecture to elite set.

        Args:
            architecture: Architecture configuration
        """
        if architecture not in self.elite_architectures:
            self.elite_architectures.append(architecture)

            # Keep only top elite architectures
            if len(self.elite_architectures) > int(self.config.elite_fraction * self.config.max_samples):
                self.elite_architectures = self.elite_architectures[-int(self.config.elite_fraction * self.config.max_samples):]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary with search statistics
        """
        return {
            'total_architectures': len(self.searched_architectures),
            'elite_architectures': len(self.elite_architectures),
            'best_score': self.best_score,
            'best_architecture_name': self.best_architecture.name if self.best_architecture else None,
            'mutations_performed': len(self.mutation_history),
            'crossovers_performed': len(self.sample_history),
            'unique_architectures': len(set(self.searched_architectures))
        }

    def reset_search(self):
        """Reset search state."""
        self.searched_architectures = []
        self.elite_architectures = []
        self.best_score = float('-inf')
        self.best_architecture = None
        self.sample_history = []
        self.mutation_history = []
        self.logger.info("🔄 Random search reset")