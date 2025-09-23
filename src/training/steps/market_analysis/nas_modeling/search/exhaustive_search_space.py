"""
Exhaustive Search Space for Complementary Model Selection

This module provides an exhaustive search space that can be used
to find complementary models for ensemble optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from itertools import combinations, product
from copy import deepcopy

from ..search.search_space import SearchSpace, ArchitectureConfig
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class ExhaustiveSearchConfig:
    """Configuration for exhaustive search space."""
    max_combinations: int = 10000  # Limit to prevent memory issues
    sample_size: int = 1000  # Sample size for large spaces
    use_sampling: bool = True  # Whether to sample from exhaustive space
    include_complementarity_constraints: bool = True
    diversity_threshold: float = 0.3  # Minimum diversity between models
    performance_threshold: float = 0.6  # Minimum individual performance

class ExhaustiveSearchSpace(SearchSpace):
    """
    Exhaustive Search Space for Complementary Models

    Provides comprehensive architecture combinations that can be
    used to select complementary models for ensemble optimization.
    """

    def __init__(self, config: ExhaustiveSearchConfig):
        """Initialize exhaustive search space.

        Args:
            config: Exhaustive search configuration
        """
        super().__init__()
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)
        self.nas_utils = NASUtils()

        # Extended search dimensions for exhaustiveness
        self.detailed_activations = [
            'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu',
            'gelu', 'swish', 'softplus', 'mish', 'hard_swish'
        ]

        self.detailed_layer_types = [
            'dense', 'conv1d', 'lstm', 'gru', 'attention',
            'residual_dense', 'residual_conv', 'dense_dropout',
            'batch_norm_dense', 'layer_norm_dense'
        ]

        self.detailed_hidden_configs = [
            [32], [64], [128], [256], [512],
            [32, 32], [64, 64], [128, 128],
            [32, 64], [64, 128], [128, 256], [256, 512],
            [32, 64, 32], [64, 128, 64], [128, 256, 128],
            [32, 64, 128, 64], [64, 128, 256, 128],
            [32, 64, 128, 256, 128]
        ]

        self.logger.info("🔍 Exhaustive search space initialized")

    def generate_exhaustive_combinations(self,
                                       input_dim: int,
                                       output_dim: int,
                                       problem_type: str = "classification",
                                       n_models: int = 3) -> List[List[ArchitectureConfig]]:
        """
        Generate exhaustive combinations of architectures for complementary models.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            problem_type: Type of problem
            n_models: Number of complementary models to find

        Returns:
            List of architecture combinations
        """
        logger.info(f"🔍 Generating exhaustive combinations for {n_models} models")

        # Generate base architectures
        base_architectures = self._generate_base_architectures(
            input_dim, output_dim, problem_type
        )

        if len(base_architectures) <= n_models:
            # Return all possible combinations if small enough
            return list(combinations(base_architectures, n_models))
        else:
            # Sample combinations if too large
            return self._sample_combinations(base_architectures, n_models)

    def _generate_base_architectures(self,
                                   input_dim: int,
                                   output_dim: int,
                                   problem_type: str) -> List[ArchitectureConfig]:
        """
        Generate comprehensive set of base architectures.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            problem_type: Type of problem

        Returns:
            List of architecture configurations
        """
        architectures = []

        # Generate all combinations of key parameters
        for hidden_config in self.detailed_hidden_configs:
            for activation in self.detailed_activations:
                for dropout in [0.0, 0.1, 0.2, 0.3]:
                    for batch_norm in [True, False]:
                        for use_residual in [True, False]:

                            # Skip invalid combinations
                            if not self._is_valid_combination(
                                hidden_config, activation, dropout, batch_norm, use_residual
                            ):
                                continue

                            # Create architecture name
                            hidden_str = '_'.join(map(str, hidden_config))
                            name = f"exhaustive_{hidden_str}_{activation}_d{dropout}_bn{batch_norm}_res{use_residual}"

                            # Create configuration
                            config = ArchitectureConfig(
                                name=name,
                                input_dim=input_dim,
                                output_dim=output_dim,
                                hidden_dims=hidden_config,
                                activation=activation,
                                dropout_rate=dropout,
                                batch_norm=batch_norm,
                                use_residual=use_residual,
                                problem_type=problem_type,
                                layer_types=['dense'] * len(hidden_config),
                                attention_heads=4,
                                embed_dim=64,
                                use_attention=False,
                                use_lstm=False,
                                use_convolution=False,
                                num_layers=len(hidden_config) + 1
                            )

                            config.calculate_complexity()
                            config.estimate_parameters()

                            architectures.append(config)

        logger.info(f"📐 Generated {len(architectures)} base architectures")
        return architectures

    def _is_valid_combination(self,
                            hidden_config: List[int],
                            activation: str,
                            dropout: float,
                            batch_norm: bool,
                            use_residual: bool) -> bool:
        """
        Check if architecture combination is valid.

        Args:
            hidden_config: Hidden layer configuration
            activation: Activation function
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections

        Returns:
            True if combination is valid
        """
        # Skip overly complex architectures
        if len(hidden_config) > 5:
            return False

        # Skip architectures that are too large
        total_params = sum(hidden_config) + hidden_config[0] * 100 + hidden_config[-1] * 5  # Rough estimate
        if total_params > 500000:  # 500k parameters limit
            return False

        # Skip invalid activation-dropout combinations
        if activation in ['sigmoid', 'tanh'] and dropout > 0.3:
            return False  # Sigmoid/tanh don't work well with high dropout

        # Skip residual without sufficient layers
        if use_residual and len(hidden_config) < 2:
            return False

        return True

    def _sample_combinations(self,
                           architectures: List[ArchitectureConfig],
                           n_models: int) -> List[List[ArchitectureConfig]]:
        """
        Sample combinations from large architecture space.

        Args:
            architectures: List of base architectures
            n_models: Number of models per combination

        Returns:
            Sampled combinations
        """
        if len(architectures) < n_models:
            return []

        n_combinations = len(architectures) * (len(architectures) - 1) * (len(architectures) - 2)
        logger.info(f"📊 Total possible combinations: {n_combinations}")

        if n_combinations <= self.config.max_combinations:
            # Generate all combinations if manageable
            return list(combinations(architectures, n_models))
        else:
            # Sample combinations
            logger.info(f"🎲 Sampling {self.config.sample_size} combinations from {n_combinations} possible")
            return self._sample_diverse_combinations(architectures, n_models)

    def _sample_diverse_combinations(self,
                                   architectures: List[ArchitectureConfig],
                                   n_models: int) -> List[List[ArchitectureConfig]]:
        """
        Sample diverse combinations to ensure complementarity.

        Args:
            architectures: List of base architectures
            n_models: Number of models per combination

        Returns:
            Diverse combination samples
        """
        sampled_combinations = []
        max_attempts = self.config.sample_size * 10  # Allow retries
        attempts = 0

        while len(sampled_combinations) < self.config.sample_size and attempts < max_attempts:
            # Randomly select n_models architectures
            selected_indices = np.random.choice(
                len(architectures), n_models, replace=False
            )
            selected_archs = [architectures[i] for i in selected_indices]

            # Check diversity
            if self._combination_meets_diversity_criteria(selected_archs):
                sampled_combinations.append(selected_archs)

            attempts += 1

        logger.info(f"✅ Sampled {len(sampled_combinations)} diverse combinations")
        return sampled_combinations

    def _combination_meets_diversity_criteria(self, architectures: List[ArchitectureConfig]) -> bool:
        """
        Check if combination meets diversity criteria.

        Args:
            architectures: List of architectures to check

        Returns:
            True if combination is sufficiently diverse
        """
        if len(architectures) < 2:
            return True

        # Calculate pairwise diversity
        min_diversity = float('inf')
        for i in range(len(architectures)):
            for j in range(i+1, len(architectures)):
                diversity = self._calculate_architecture_diversity(
                    architectures[i], architectures[j]
                )
                min_diversity = min(min_diversity, diversity)

        return min_diversity >= self.config.diversity_threshold

    def _calculate_architecture_diversity(self, arch1: ArchitectureConfig, arch2: ArchitectureConfig) -> float:
        """
        Calculate diversity between two architectures.

        Args:
            arch1: First architecture
            arch2: Second architecture

        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        diversity = 0.0
        factors = 0

        # Hidden dimensions diversity
        if arch1.hidden_dims != arch2.hidden_dims:
            diversity += 1.0
        factors += 1

        # Activation diversity
        if arch1.activation != arch2.activation:
            diversity += 1.0
        factors += 1

        # Dropout diversity
        dropout_diff = abs(arch1.dropout_rate - arch2.dropout_rate)
        diversity += min(dropout_diff * 2, 1.0)
        factors += 1

        # Architecture features diversity
        features = ['batch_norm', 'use_residual', 'use_attention', 'use_lstm', 'use_convolution']
        for feature in features:
            val1 = getattr(arch1, feature, False)
            val2 = getattr(arch2, feature, False)
            if val1 != val2:
                diversity += 1.0
            factors += 1

        # Complexity diversity
        complexity_diff = abs(arch1.complexity_score - arch2.complexity_score)
        diversity += min(complexity_diff / 5.0, 1.0)  # Scale to 0-1
        factors += 1

        return diversity / factors if factors > 0 else 0.0

    def find_complementary_ensembles(self,
                                   architectures: List[ArchitectureConfig],
                                   n_models: int = 3,
                                   max_ensembles: int = 100) -> List[List[ArchitectureConfig]]:
        """
        Find complementary architecture ensembles.

        Args:
            architectures: List of available architectures
            n_models: Number of models per ensemble
            max_ensembles: Maximum number of ensembles to return

        Returns:
            List of complementary architecture combinations
        """
        logger.info(f"🔍 Finding complementary ensembles of {n_models} models")

        if len(architectures) < n_models:
            return []

        # Generate all possible combinations
        all_combinations = list(combinations(architectures, n_models))

        if len(all_combinations) <= max_ensembles:
            # Score all combinations
            scored_combinations = []
            for combo in all_combinations:
                score = self._evaluate_ensemble_potential(combo)
                scored_combinations.append((combo, score))

            # Sort by score and return top ensembles
            scored_combinations.sort(key=lambda x: x[1], reverse=True)
            top_ensembles = [combo for combo, score in scored_combinations[:max_ensembles]]

        else:
            # Sample and evaluate
            sampled_combinations = []
            for _ in range(max_ensembles):
                # Sample combination
                indices = np.random.choice(len(all_combinations), 1, replace=False)[0]
                combo = all_combinations[indices]

                # Evaluate potential
                potential = self._evaluate_ensemble_potential(combo)
                sampled_combinations.append((combo, potential))

            # Sort and return
            sampled_combinations.sort(key=lambda x: x[1], reverse=True)
            top_ensembles = [combo for combo, score in sampled_combinations]

        logger.info(f"✅ Found {len(top_ensembles)} complementary ensembles")
        return top_ensembles

    def _evaluate_ensemble_potential(self, architectures: List[ArchitectureConfig]) -> float:
        """
        Evaluate the potential of an architecture ensemble.

        Args:
            architectures: List of architectures in ensemble

        Returns:
            Ensemble potential score
        """
        if len(architectures) < 2:
            return 0.0

        # Individual performance potential (based on complexity)
        individual_scores = []
        for arch in architectures:
            score = self._estimate_individual_performance(arch)
            individual_scores.append(score)

        avg_individual = np.mean(individual_scores)

        # Complementarity score
        complementarity = 0.0
        n_pairs = 0

        for i in range(len(architectures)):
            for j in range(i+1, len(architectures)):
                diversity = self._calculate_architecture_diversity(
                    architectures[i], architectures[j]
                )
                complementarity += diversity
                n_pairs += 1

        avg_complementarity = complementarity / n_pairs if n_pairs > 0 else 0.0

        # Ensemble potential combines individual performance and complementarity
        ensemble_potential = 0.7 * avg_individual + 0.3 * avg_complementarity

        return ensemble_potential

    def _estimate_individual_performance(self, architecture: ArchitectureConfig) -> float:
        """
        Estimate individual architecture performance.

        Args:
            architecture: Architecture to evaluate

        Returns:
            Estimated performance score
        """
        # Simple estimation based on architecture properties
        performance = 0.0

        # Base performance from architecture size
        total_neurons = sum(architecture.hidden_dims) + architecture.input_dim + architecture.output_dim
        performance += min(total_neurons / 1000.0, 1.0) * 0.4

        # Bonus for good practices
        if architecture.batch_norm:
            performance += 0.1

        if architecture.use_residual and len(architecture.hidden_dims) >= 2:
            performance += 0.1

        if architecture.dropout_rate > 0 and architecture.dropout_rate <= 0.3:
            performance += 0.05

        # Penalty for too complex architectures
        if architecture.complexity_score > 5.0:
            performance -= 0.2

        # Penalty for too many parameters
        if architecture.estimated_params > 100000:
            performance -= 0.1

        return max(0.0, min(1.0, performance))

    def get_search_space_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the exhaustive search space.

        Returns:
            Dictionary with search space statistics
        """
        n_base_architectures = len(self._generate_base_architectures(100, 5, "classification"))
        n_activations = len(self.detailed_activations)
        n_hidden_configs = len(self.detailed_hidden_configs)
        n_dropout_options = 4  # 0.0, 0.1, 0.2, 0.3
        n_bool_options = 2  # True/False for batch_norm, use_residual

        total_combinations = n_base_architectures * n_activations * n_dropout_options * n_bool_options

        return {
            'base_architectures': n_base_architectures,
            'activation_functions': n_activations,
            'hidden_configurations': n_hidden_configs,
            'dropout_options': n_dropout_options,
            'boolean_options': n_bool_options,
            'total_combinations': total_combinations,
            'is_exhaustive': total_combinations <= self.config.max_combinations,
            'max_combinations_limit': self.config.max_combinations,
            'would_use_sampling': total_combinations > self.config.sample_size
        }

    def optimize_ensemble_selection(self,
                                  architectures: List[ArchitectureConfig],
                                  n_models: int = 3) -> List[ArchitectureConfig]:
        """
        Optimize selection of complementary models from architecture list.

        Args:
            architectures: List of available architectures
            n_models: Number of models to select

        Returns:
            Optimal complementary model selection
        """
        logger.info(f"⚖️ Optimizing ensemble selection of {n_models} models")

        # Find complementary ensembles
        complementary_ensembles = self.find_complementary_ensembles(
            architectures, n_models, max_ensembles=50
        )

        if not complementary_ensembles:
            # Fallback to random selection
            logger.warning("⚠️ No complementary ensembles found, using random selection")
            return list(np.random.choice(architectures, n_models, replace=False))

        # Return the best ensemble (first one)
        best_ensemble = complementary_ensembles[0]

        logger.info(f"✅ Selected optimal ensemble with {len(best_ensemble)} models")
        for i, model in enumerate(best_ensemble):
            logger.info(f"   Model {i+1}: {model.name} (complexity: {model.complexity_score:.3f})")

        return best_ensemble