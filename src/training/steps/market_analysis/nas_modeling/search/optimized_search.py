"""
Optimized Search Strategies with Grid Utilities Integration

This module provides optimized search strategies that integrate with
the existing grid utilities for more efficient parameter optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy
import time

from ..search.search_space import SearchSpace, ArchitectureConfig
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

# Import grid utilities
try:
    from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
    GRID_UTILS_AVAILABLE = True
except ImportError:
    GRID_UTILS_AVAILABLE = False
    build_coarse_grid_from_search_space = None
    build_fine_grid_around_best = None

logger = logging.getLogger(__name__)

@dataclass
class OptimizedSearchConfig:
    """Configuration for optimized search strategies."""
    use_grid_integration: bool = True
    grid_points: int = 10
    two_step_optimization: bool = True
    adaptive_sampling: bool = True
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    random_seed: int = 42

class OptimizedRandomSearch:
    """
    Optimized Random Search with Grid Integration

    Integrates with grid utilities for more efficient exploration
    and uses adaptive sampling for better convergence.
    """

    def __init__(self, config: OptimizedSearchConfig):
        """Initialize optimized random search.

        Args:
            config: Optimized search configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Initialize components
        self.search_space = SearchSpace()
        self.nas_utils = NASUtils()

        # Search state
        self.searched_architectures = []
        self.best_architecture = None
        self.best_score = float('-inf')
        self.grid_history = []

        # Adaptive sampling state
        self.successful_architectures = []
        self.mutation_history = []

        self.logger.info("🚀 Optimized Random Search initialized")

    def generate_architecture(self, iteration: int) -> Optional[ArchitectureConfig]:
        """
        Generate architecture using optimized random search.

        Args:
            iteration: Current search iteration

        Returns:
            Optimized architecture configuration
        """
        try:
            # Use grid integration if enabled and available
            if (self.config.use_grid_integration and GRID_UTILS_AVAILABLE
                and iteration % 5 == 0 and self.successful_architectures):
                return self._generate_grid_optimized_architecture(iteration)
            elif self.config.adaptive_sampling and len(self.successful_architectures) >= 3:
                return self._generate_adaptive_architecture(iteration)
            else:
                return self._generate_random_architecture()

        except Exception as e:
            self.logger.error(f"❌ Failed to generate architecture at iteration {iteration}: {e}")
            return None

    def _generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random architecture.

        Returns:
            Random architecture configuration
        """
        input_dim = 100  # Default for market data
        output_dim = 5   # Default for regime detection

        architecture = self.search_space.generate_random_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            problem_type="regime_detection"
        )

        return architecture

    def _generate_grid_optimized_architecture(self, iteration: int) -> ArchitectureConfig:
        """Generate architecture using grid optimization.

        Args:
            iteration: Current search iteration

        Returns:
            Grid-optimized architecture configuration
        """
        if not GRID_UTILS_AVAILABLE:
            return self._generate_random_architecture()

        # Define search space for grid utilities
        search_space = self._build_search_space_for_grid()

        # Build coarse grid around best architectures
        if self.successful_architectures:
            best_arch = self._get_best_architecture()
            best_params = self._architecture_to_grid_params(best_arch)

            # Create fine grid around best
            grid_params = build_fine_grid_around_best(
                search_space, best_params, self.config.grid_points
            )
        else:
            # Create coarse grid
            grid_params = build_coarse_grid_from_search_space(
                search_space, self.config.grid_points
            )

        if not grid_params:
            return self._generate_random_architecture()

        # Select random parameter set from grid
        selected_params = self.nas_utils.random_choice(grid_params)
        architecture = self._grid_params_to_architecture(selected_params)

        self.logger.debug(f"🔍 Generated grid-optimized architecture: {architecture.name}")
        self.grid_history.append(architecture.name)

        return architecture

    def _generate_adaptive_architecture(self, iteration: int) -> ArchitectureConfig:
        """Generate architecture using adaptive sampling.

        Args:
            iteration: Current search iteration

        Returns:
            Adaptive architecture configuration
        """
        # Choose adaptive strategy
        strategy = self._select_adaptive_strategy(iteration)

        if strategy == "mutation" and len(self.successful_architectures) >= 2:
            return self._mutate_successful_architecture()
        elif strategy == "crossover" and len(self.successful_architectures) >= 3:
            return self._crossover_successful_architectures()
        else:
            # Fallback to random with bias toward successful architectures
            if self.successful_architectures and self.nas_utils.random_float() < 0.7:
                return deepcopy(self.nas_utils.random_choice(self.successful_architectures))
            else:
                return self._generate_random_architecture()

    def _select_adaptive_strategy(self, iteration: int) -> str:
        """Select adaptive strategy based on iteration and success.

        Args:
            iteration: Current search iteration

        Returns:
            Strategy name
        """
        if len(self.successful_architectures) < 2:
            return "random"

        if iteration % 10 == 0:  # Every 10 iterations
            return "mutation"
        elif len(self.successful_architectures) >= 3:
            return "crossover"
        else:
            return "random"

    def _mutate_successful_architecture(self) -> ArchitectureConfig:
        """Mutate a successful architecture.

        Returns:
            Mutated architecture configuration
        """
        if not self.successful_architectures:
            return self._generate_random_architecture()

        # Select successful parent
        parent = self.nas_utils.random_choice(self.successful_architectures)

        # Create mutation with slight variations
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

        # Apply small mutations
        mutations_applied = 0

        # Small changes to hidden dimensions
        if mutated_config.hidden_dims and self.nas_utils.random_float() < 0.5:
            idx = self.nas_utils.random_int(0, len(mutated_config.hidden_dims) - 1)
            change_factor = 1.0 + self.nas_utils.random_float(-0.2, 0.2)  # ±20%
            new_dim = max(16, int(mutated_config.hidden_dims[idx] * change_factor))
            mutated_config.hidden_dims[idx] = new_dim
            mutations_applied += 1

        # Small change to dropout
        if self.nas_utils.random_float() < 0.3:
            change_factor = 1.0 + self.nas_utils.random_float(-0.1, 0.1)  # ±10%
            new_dropout = max(0.0, min(0.5, mutated_config.dropout_rate * change_factor))
            mutated_config.dropout_rate = round(new_dropout, 2)
            mutations_applied += 1

        # Toggle features with low probability
        if self.nas_utils.random_float() < 0.1:
            mutated_config.batch_norm = not mutated_config.batch_norm
            mutations_applied += 1

        if self.nas_utils.random_float() < 0.1:
            mutated_config.use_residual = not mutated_config.use_residual
            mutations_applied += 1

        # Recalculate metrics
        mutated_config.calculate_complexity()
        mutated_config.estimate_parameters()

        self.logger.debug(f"🧬 Applied {mutations_applied} mutations to {parent.name}")
        self.mutation_history.append(mutated_config.name)

        return mutated_config

    def _crossover_successful_architectures(self) -> ArchitectureConfig:
        """Create architecture by crossing over successful architectures.

        Returns:
            Crossover architecture configuration
        """
        if len(self.successful_architectures) < 2:
            return self.nas_utils.random_choice(self.successful_architectures)

        # Select two successful parents
        parent1 = self.nas_utils.random_choice(self.successful_architectures)
        parent2 = self.nas_utils.random_choice(self.successful_architectures)

        # Create offspring with weighted inheritance
        offspring = ArchitectureConfig(
            name=f"{parent1.name}_x_{parent2.name}_{len(self.grid_history)}",
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

        # Crossover with 70% from parent1, 30% from parent2
        if self.nas_utils.random_float() < 0.3:
            offspring.hidden_dims = parent2.hidden_dims.copy()

        if self.nas_utils.random_float() < 0.2:
            offspring.activation = parent2.activation

        if self.nas_utils.random_float() < 0.2:
            offspring.dropout_rate = parent2.dropout_rate

        if self.nas_utils.random_float() < 0.15:
            offspring.batch_norm = parent2.batch_norm

        if self.nas_utils.random_float() < 0.15:
            offspring.use_residual = parent2.use_residual

        # Recalculate metrics
        offspring.calculate_complexity()
        offspring.estimate_parameters()

        self.logger.debug(f"🔄 Created crossover from {parent1.name} and {parent2.name}")
        self.grid_history.append(offspring.name)

        return offspring

    def _build_search_space_for_grid(self) -> Dict[str, Any]:
        """Build search space compatible with grid utilities.

        Returns:
            Search space dictionary
        """
        return {
            'n_layers': {'type': 'int', 'low': 1, 'high': 5},
            'hidden_units': {'type': 'int', 'low': 16, 'high': 512},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'leaky_relu', 'elu']},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'batch_norm': {'type': 'categorical', 'choices': [True, False]},
            'use_residual': {'type': 'categorical', 'choices': [True, False]},
            'use_attention': {'type': 'categorical', 'choices': [True, False]},
            'attention_heads': {'type': 'int', 'low': 2, 'high': 16}
        }

    def _architecture_to_grid_params(self, architecture: ArchitectureConfig) -> Dict[str, Any]:
        """Convert architecture to grid parameters.

        Args:
            architecture: Architecture configuration

        Returns:
            Grid parameters dictionary
        """
        return {
            'n_layers': len(architecture.hidden_dims) + 1,  # +1 for output layer
            'hidden_units': architecture.hidden_dims[0] if architecture.hidden_dims else 64,
            'activation': architecture.activation,
            'dropout': architecture.dropout_rate,
            'batch_norm': architecture.batch_norm,
            'use_residual': architecture.use_residual,
            'use_attention': architecture.use_attention,
            'attention_heads': architecture.attention_heads
        }

    def _grid_params_to_architecture(self, params: Dict[str, Any]) -> ArchitectureConfig:
        """Convert grid parameters to architecture.

        Args:
            params: Grid parameters

        Returns:
            Architecture configuration
        """
        # Create hidden dimensions based on number of layers
        n_layers = params['n_layers'] - 1  # -1 for output layer
        hidden_units = params['hidden_units']
        hidden_dims = [hidden_units] * max(1, n_layers)

        config = ArchitectureConfig(
            name=f"grid_opt_{len(self.grid_history)}",
            input_dim=100,
            output_dim=5,
            hidden_dims=hidden_dims,
            activation=params['activation'],
            dropout_rate=params['dropout'],
            batch_norm=params['batch_norm'],
            use_residual=params['use_residual'],
            problem_type="regime_detection",
            use_attention=params['use_attention'],
            attention_heads=params['attention_heads']
        )

        config.calculate_complexity()
        config.estimate_parameters()

        return config

    def _get_best_architecture(self) -> ArchitectureConfig:
        """Get the best architecture found so far.

        Returns:
            Best architecture configuration
        """
        if not self.successful_architectures:
            return self._generate_random_architecture()

        # Return architecture with highest score (placeholder - would need scores)
        return self.successful_architectures[0]

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

    def add_successful_architecture(self, architecture: ArchitectureConfig):
        """Add architecture to successful architectures list.

        Args:
            architecture: Successful architecture
        """
        if architecture not in self.successful_architectures:
            self.successful_architectures.append(architecture)

            # Keep only top architectures
            if len(self.successful_architectures) > 10:
                self.successful_architectures = self.successful_architectures[-10:]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary with search statistics
        """
        return {
            'total_architectures': len(self.searched_architectures),
            'successful_architectures': len(self.successful_architectures),
            'best_score': self.best_score,
            'grid_optimizations': len(self.grid_history),
            'mutations_performed': len(self.mutation_history),
            'grid_utils_available': GRID_UTILS_AVAILABLE
        }

    def reset_search(self):
        """Reset search state."""
        self.searched_architectures = []
        self.successful_architectures = []
        self.best_architecture = None
        self.best_score = float('-inf')
        self.grid_history = []
        self.mutation_history = []
        self.logger.info("🔄 Optimized random search reset")


class OptimizedBayesianSearch:
    """
    Optimized Bayesian Search with Grid Integration

    Uses grid utilities for initial exploration and Bayesian optimization
    for fine-tuning around promising regions.
    """

    def __init__(self, config: OptimizedSearchConfig):
        """Initialize optimized Bayesian search.

        Args:
            config: Optimized search configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Initialize components
        self.search_space = SearchSpace()
        self.nas_utils = NASUtils()

        # Search state
        self.X_train = []  # Architecture encodings
        self.y_train = []  # Architecture scores
        self.searched_architectures = []

        # Grid integration state
        self.grid_exploration_done = False
        self.best_grid_params = None

        # Simple GP model parameters (placeholder for actual GP)
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = 0.01

        self.logger.info("🧠 Optimized Bayesian Search initialized")

    def generate_architecture(self, iteration: int) -> Optional[ArchitectureConfig]:
        """
        Generate architecture using optimized Bayesian search.

        Args:
            iteration: Current search iteration

        Returns:
            Optimized architecture configuration
        """
        try:
            # Initial grid exploration
            if (self.config.use_grid_integration and GRID_UTILS_AVAILABLE
                and not self.grid_exploration_done and iteration <= 10):
                return self._generate_grid_exploration_architecture(iteration)
            elif iteration <= 5:  # Initial random samples
                return self._generate_random_architecture()
            else:
                return self._generate_bayesian_architecture()

        except Exception as e:
            self.logger.error(f"❌ Failed to generate architecture at iteration {iteration}: {e}")
            return None

    def _generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random architecture for initial sampling.

        Returns:
            Random architecture configuration
        """
        input_dim = 100
        output_dim = 5

        architecture = self.search_space.generate_random_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            problem_type="regime_detection"
        )

        return architecture

    def _generate_grid_exploration_architecture(self, iteration: int) -> ArchitectureConfig:
        """Generate architecture using grid exploration.

        Args:
            iteration: Current search iteration

        Returns:
            Grid exploration architecture
        """
        if not GRID_UTILS_AVAILABLE:
            return self._generate_random_architecture()

        # Build search space for grid
        search_space = self._build_search_space_for_grid()

        # Create grid
        if self.best_grid_params is None:
            # First grid - coarse exploration
            grid_params = build_coarse_grid_from_search_space(search_space, self.config.grid_points)
        else:
            # Subsequent grids - fine around best
            grid_params = build_fine_grid_around_best(search_space, self.best_grid_params, self.config.grid_points)

        if not grid_params or iteration > len(grid_params):
            self.grid_exploration_done = True
            return self._generate_random_architecture()

        selected_params = grid_params[iteration - 1]
        architecture = self._grid_params_to_architecture(selected_params)

        self.logger.debug(f"🔍 Generated grid exploration architecture: {architecture.name}")
        return architecture

    def _generate_bayesian_architecture(self) -> ArchitectureConfig:
        """Generate architecture using Bayesian optimization.

        Returns:
            Bayesian-optimized architecture
        """
        if len(self.X_train) < 2:
            return self._generate_random_architecture()

        # Use grid utilities to optimize around best region
        if (self.config.use_grid_integration and GRID_UTILS_AVAILABLE
            and self.best_grid_params is not None):
            return self._generate_grid_bayesian_architecture()
        else:
            return self._generate_simple_bayesian_architecture()

    def _generate_grid_bayesian_architecture(self) -> ArchitectureConfig:
        """Generate architecture using grid + Bayesian optimization.

        Returns:
            Grid-Bayesian optimized architecture
        """
        if not GRID_UTILS_AVAILABLE:
            return self._generate_simple_bayesian_architecture()

        # Build search space
        search_space = self._build_search_space_for_grid()

        # Create fine grid around current best
        grid_params = build_fine_grid_around_best(search_space, self.best_grid_params, 20)

        if not grid_params:
            return self._generate_simple_bayesian_architecture()

        # Select best from grid using simple acquisition function
        best_idx = self._select_best_grid_candidate(grid_params)
        selected_params = grid_params[best_idx]

        architecture = self._grid_params_to_architecture(selected_params)
        self.logger.debug(f"🧠 Generated grid-Bayesian architecture: {architecture.name}")

        return architecture

    def _generate_simple_bayesian_architecture(self) -> ArchitectureConfig:
        """Generate architecture using simple Bayesian optimization.

        Returns:
            Simple Bayesian architecture
        """
        # For now, use random selection with bias toward better architectures
        # In a full implementation, this would use proper GP modeling

        if not self.X_train:
            return self._generate_random_architecture()

        # Simple strategy: generate architecture close to best performing ones
        best_indices = np.argsort(self.y_train)[-3:]  # Top 3 architectures

        if len(best_indices) > 0:
            best_idx = self.nas_utils.random_choice(best_indices)
            best_arch = self.searched_architectures[best_idx]

            # Create variation of best architecture
            varied_config = ArchitectureConfig(
                name=f"bayes_var_{len(self.searched_architectures)}",
                input_dim=best_arch.input_dim,
                output_dim=best_arch.output_dim,
                hidden_dims=best_arch.hidden_dims.copy(),
                activation=best_arch.activation,
                dropout_rate=max(0.0, min(0.5, best_arch.dropout_rate + self.nas_utils.random_float(-0.1, 0.1))),
                batch_norm=best_arch.batch_norm,
                use_residual=best_arch.use_residual,
                problem_type=best_arch.problem_type
            )

            varied_config.calculate_complexity()
            varied_config.estimate_parameters()

            return varied_config
        else:
            return self._generate_random_architecture()

    def _build_search_space_for_grid(self) -> Dict[str, Any]:
        """Build search space for grid utilities.

        Returns:
            Search space dictionary
        """
        return {
            'hidden_units': {'type': 'int', 'low': 32, 'high': 512},
            'n_layers': {'type': 'int', 'low': 2, 'high': 6},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'leaky_relu', 'elu']},
            'batch_norm': {'type': 'categorical', 'choices': [True, False]},
            'use_residual': {'type': 'categorical', 'choices': [True, False]}
        }

    def _grid_params_to_architecture(self, params: Dict[str, Any]) -> ArchitectureConfig:
        """Convert grid parameters to architecture.

        Args:
            params: Grid parameters

        Returns:
            Architecture configuration
        """
        n_layers = params['n_layers'] - 1  # -1 for output layer
        hidden_dims = [params['hidden_units']] * max(1, n_layers)

        config = ArchitectureConfig(
            name=f"grid_bayes_{len(self.searched_architectures)}",
            input_dim=100,
            output_dim=5,
            hidden_dims=hidden_dims,
            activation=params['activation'],
            dropout_rate=params['dropout'],
            batch_norm=params['batch_norm'],
            use_residual=params['use_residual'],
            problem_type="regime_detection"
        )

        config.calculate_complexity()
        config.estimate_parameters()

        return config

    def _select_best_grid_candidate(self, grid_params: List[Dict[str, Any]]) -> int:
        """Select best candidate from grid using simple acquisition.

        Args:
            grid_params: List of grid parameter sets

        Returns:
            Index of best candidate
        """
        if len(self.X_train) < 2:
            return 0

        best_idx = 0
        best_acquisition = float('-inf')

        for i, params in enumerate(grid_params):
            # Simple acquisition: distance to best architectures + randomness
            acquisition = self._calculate_simple_acquisition(params)

            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_idx = i

        return best_idx

    def _calculate_simple_acquisition(self, params: Dict[str, Any]) -> float:
        """Calculate simple acquisition function value.

        Args:
            params: Grid parameters

        Returns:
            Acquisition value
        """
        if not self.X_train:
            return 0.0

        # Find closest architecture in training data
        min_distance = float('inf')

        for i, arch_encoding in enumerate(self.X_train):
            distance = self._calculate_parameter_distance(params, arch_encoding)
            min_distance = min(min_distance, distance)

        # Acquisition combines distance (exploration) and score (exploitation)
        best_score = max(self.y_train) if self.y_train else 0.0
        acquisition = best_score * 0.5 + (1.0 / (1.0 + min_distance)) * 0.5

        return acquisition

    def _calculate_parameter_distance(self, params1: Dict[str, Any], params2: np.ndarray) -> float:
        """Calculate distance between parameter sets.

        Args:
            params1: Grid parameters
            params2: Architecture encoding

        Returns:
            Distance value
        """
        # Simplified distance calculation
        distance = 0.0

        # Hidden units distance
        if 'hidden_units' in params1:
            hidden_units_idx = 0  # Simplified - would need proper encoding mapping
            distance += abs(params1['hidden_units'] - params2[hidden_units_idx])

        # Dropout distance
        if 'dropout' in params1:
            dropout_idx = 1  # Simplified
            distance += abs(params1['dropout'] - params2[dropout_idx])

        return distance

    def update_observations(self, architecture: ArchitectureConfig, score: float):
        """Update observations with new architecture and score.

        Args:
            architecture: Evaluated architecture
            score: Architecture score
        """
        encoding = self._encode_architecture(architecture)
        self.X_train.append(encoding)
        self.y_train.append(score)
        self.searched_architectures.append(architecture)

        # Update best grid params if this is better
        if score > self.best_score:
            self.best_score = score
            self.best_grid_params = self._architecture_to_grid_params(architecture)

        self.logger.debug(f"📊 Added observation: {architecture.name} -> {score:.4f}")

    def _encode_architecture(self, architecture: ArchitectureConfig) -> np.ndarray:
        """Encode architecture to numerical vector.

        Args:
            architecture: Architecture configuration

        Returns:
            Numerical encoding
        """
        encoding = []

        # Hidden dimensions
        hidden_dims = architecture.hidden_dims[:3] + [0] * max(0, 3 - len(architecture.hidden_dims))
        encoding.extend(hidden_dims)

        # Activation (one-hot simplified)
        activations = ['relu', 'tanh', 'leaky_relu', 'elu']
        activation_idx = activations.index(architecture.activation) if architecture.activation in activations else 0
        encoding.append(float(activation_idx))

        # Dropout
        encoding.append(architecture.dropout_rate)

        # Boolean features
        encoding.extend([
            1.0 if architecture.batch_norm else 0.0,
            1.0 if architecture.use_residual else 0.0,
            1.0 if architecture.use_attention else 0.0,
            architecture.complexity_score,
            architecture.estimated_params / 1000000.0
        ])

        return np.array(encoding)

    def _architecture_to_grid_params(self, architecture: ArchitectureConfig) -> Dict[str, Any]:
        """Convert architecture to grid parameters.

        Args:
            architecture: Architecture configuration

        Returns:
            Grid parameters
        """
        return {
            'hidden_units': architecture.hidden_dims[0] if architecture.hidden_dims else 64,
            'n_layers': len(architecture.hidden_dims) + 1,
            'dropout': architecture.dropout_rate,
            'activation': architecture.activation,
            'batch_norm': architecture.batch_norm,
            'use_residual': architecture.use_residual
        }

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary with search statistics
        """
        return {
            'n_observations': len(self.X_train),
            'best_score': max(self.y_train) if self.y_train else None,
            'grid_exploration_done': self.grid_exploration_done,
            'best_grid_params': self.best_grid_params,
            'grid_utils_available': GRID_UTILS_AVAILABLE
        }

    def reset_search(self):
        """Reset search state."""
        self.X_train = []
        self.y_train = []
        self.searched_architectures = []
        self.grid_exploration_done = False
        self.best_grid_params = None
        self.best_score = float('-inf')
        self.logger.info("🔄 Optimized Bayesian search reset")