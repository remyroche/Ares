"""
Advanced Search for TAS Tree Architecture

This module provides advanced search strategies for tree architecture search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdvancedSearchConfig:
    """Configuration for advanced search."""
    n_iterations: int = 1000
    population_size: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    diversity_threshold: float = 0.1
    convergence_threshold: float = 0.01


class AdvancedTASSearch:
    """Advanced search for tree architectures."""

    def __init__(self, config: AdvancedSearchConfig):
        self.original_config = config
        self.config = self._build_search_config(config)
        self.population: List[Dict[str, Any]] = []
        self.fitness_scores: List[float] = []
        self.best_individuals: List[Dict[str, Any]] = []
        self.best_individual: Optional[Dict[str, Any]] = None
        self.best_score: float = float("-inf")
        self.best_score_history: List[float] = []
        self.generation = 0
        self.market_data: Optional[pd.DataFrame] = None
        self.target_returns: Optional[pd.Series] = None
        self.market_regimes: Dict[str, Any] = {}
        self.micro_regimes: Dict[str, Any] = {}
        self.architecture_type: Optional[Any] = None

    def search(
        self,
        *,
        market_data: pd.DataFrame,
        target_returns: pd.Series,
        market_regimes: Optional[Dict[str, Any]] = None,
        micro_regimes: Optional[Dict[str, Any]] = None,
        architecture_type: Optional[Any] = None,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform advanced search for optimal tree architecture."""
        logger.info("Starting advanced TAS search")

        if market_data is None or target_returns is None:
            raise ValueError("market_data and target_returns must be provided for advanced search")

        self.market_data = market_data
        self.target_returns = target_returns
        self.market_regimes = market_regimes or {}
        self.micro_regimes = micro_regimes or {}
        self.architecture_type = architecture_type
        self.best_individual = None
        self.best_score = float("-inf")
        self.best_score_history = []
        self.previous_best_fitness = None

        if search_space is None or len(search_space) == 0:
            search_space = self._derive_default_search_space(market_data)

        if not search_space:
            raise ValueError("A non-empty search_space is required for advanced TAS search")

        # Initialize population
        self._initialize_population(search_space)
        
        # Evolution loop
        final_generation = 0
        for generation in range(self.config.n_iterations):
            self.generation = generation
            final_generation = generation

            # Evaluate fitness
            self._evaluate_population()

            # Update best individuals
            self._update_best_individuals()
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged at generation {generation}")
                break
            
            # Selection
            parents = self._select_parents()
            
            # Crossover and mutation
            offspring = self._create_offspring(parents, search_space)
            
            # Update population
            self._update_population(offspring)
            
            if generation % 100 == 0:
                logger.info(f"Generation {generation} completed")

        # Determine best architecture and score
        if self.best_individual is None and self.population:
            # Evaluate current population to identify best candidate
            self._evaluate_population()
            self._update_best_individuals()

        result = {
            'best_architecture': self.best_individual.copy() if self.best_individual else {},
            'best_score': self.best_score if self.best_individual else None,
            'best_score_history': list(self.best_score_history),
            'generations_evaluated': final_generation + 1,
            'population': [ind.copy() for ind in self.population],
            'fitness_scores': list(self.fitness_scores),
        }

        return result

    def _build_search_config(self, config: AdvancedSearchConfig) -> AdvancedSearchConfig:
        """Create a consistent search configuration from various config inputs."""

        def _get(attr: str, default: Any) -> Any:
            if hasattr(config, attr):
                return getattr(config, attr)
            # Allow TASConfig style attribute names
            fallback_map = {
                'n_iterations': 'n_search_iterations',
                'population_size': 'population_size',
                'mutation_rate': 'mutation_rate',
                'crossover_rate': 'crossover_rate',
                'elite_size': 'elite_size',
                'diversity_threshold': 'diversity_threshold',
                'convergence_threshold': 'convergence_threshold',
            }
            fallback_name = fallback_map.get(attr)
            if fallback_name and hasattr(config, fallback_name):
                return getattr(config, fallback_name)
            return default

        elite_default = min(_get('population_size', AdvancedSearchConfig.population_size), AdvancedSearchConfig.elite_size)

        return AdvancedSearchConfig(
            n_iterations=int(_get('n_iterations', AdvancedSearchConfig.n_iterations)),
            population_size=int(_get('population_size', AdvancedSearchConfig.population_size)),
            elite_size=int(_get('elite_size', elite_default)),
            mutation_rate=float(_get('mutation_rate', AdvancedSearchConfig.mutation_rate)),
            crossover_rate=float(_get('crossover_rate', AdvancedSearchConfig.crossover_rate)),
            diversity_threshold=float(_get('diversity_threshold', AdvancedSearchConfig.diversity_threshold)),
            convergence_threshold=float(_get('convergence_threshold', AdvancedSearchConfig.convergence_threshold)),
        )
    
    def _initialize_population(self, search_space: Dict[str, Any]):
        """Initialize population with random individuals."""
        self.population = []
        for _ in range(self.config.population_size):
            individual = self._create_random_individual(search_space)
            self.population.append(individual)

    def _create_random_individual(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create a random individual from search space."""
        individual = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                individual[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Range parameter
                individual[param] = np.random.uniform(values[0], values[1])
            else:
                individual[param] = values
        return individual
    
    def _evaluate_population(self):
        """Evaluate fitness of all individuals in population."""
        self.fitness_scores = []
        for individual in self.population:
            fitness = self._evaluate_individual(individual)
            self.fitness_scores.append(fitness)

    def _evaluate_individual(self, individual: Dict[str, Any]) -> float:
        """Evaluate fitness of a single individual."""
        if self.market_data is None or self.target_returns is None:
            raise ValueError("Market data and target returns must be set before evaluation")

        numeric_data = self.market_data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return float('-inf')

        target = self.target_returns
        if isinstance(target, pd.DataFrame):
            target = target.iloc[:, 0]
        target = target.reindex(numeric_data.index)
        target = target.astype(float)
        valid_mask = target.notna()
        if not valid_mask.any():
            return float('-inf')

        X = numeric_data.loc[valid_mask].to_numpy(dtype=float)
        y = target.loc[valid_mask].to_numpy(dtype=float)
        if X.size == 0:
            return float('-inf')

        weight_vector = self._individual_to_weight_vector(individual, X.shape[1])
        predictions = X @ weight_vector

        pred_std = float(np.std(predictions))
        target_std = float(np.std(y))
        if pred_std < 1e-12 or target_std < 1e-12:
            # Degenerate case - reward proximity to target mean
            mse = float(np.mean((y - np.mean(y)) ** 2))
            return -mse

        correlation = float(np.corrcoef(predictions, y)[0, 1])
        mse = float(np.mean((predictions - y) ** 2))
        sharpe_like = float(np.mean(predictions) / (pred_std + 1e-8))

        # Higher correlation and Sharpe, lower error desired
        score = correlation + 0.05 * sharpe_like - mse
        return score

    def _update_best_individuals(self):
        """Update best individuals based on fitness."""
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]

        # Update best individuals
        self.best_individuals = [self.population[i] for i in sorted_indices[:self.config.elite_size]]

        if sorted_indices.size == 0:
            return

        best_idx = sorted_indices[0]
        best_score = self.fitness_scores[best_idx]
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_individual = self.population[best_idx].copy()
        self.best_score_history.append(self.best_score)

    def _check_convergence(self) -> bool:
        """Check if the population has converged."""
        if len(self.fitness_scores) < 2:
            return False
        
        # Check if fitness improvement is below threshold
        best_fitness = max(self.fitness_scores)
        previous_best = getattr(self, 'previous_best_fitness', None)
        if previous_best is not None:
            improvement = abs(best_fitness - previous_best)
            if improvement < self.config.convergence_threshold:
                return True

        self.previous_best_fitness = best_fitness
        return False
    
    def _select_parents(self) -> List[Dict[str, Any]]:
        """Select parents for reproduction."""
        # Tournament selection
        parents = []
        for _ in range(self.config.population_size):
            tournament_size = 3
            tournament_indices = np.random.choice(
                len(self.population), tournament_size, replace=False
            )
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        return parents
    
    def _create_offspring(self, parents: List[Dict[str, Any]], search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create offspring through crossover and mutation."""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parents[i], parents[i + 1]])
            else:
                offspring.append(parents[i])

        # Mutation
        for child in offspring:
            if np.random.random() < self.config.mutation_rate:
                self._mutate(child, search_space)

        return offspring
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        keys = list(parent1.keys())
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], search_space: Dict[str, Any]):
        """Mutate an individual."""
        for key, value in individual.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(value, (int, float)):
                    # Add small random change
                    noise = np.random.normal(0, 0.1 * abs(value))
                    individual[key] = value + noise
                elif isinstance(value, str):
                    # Random choice from possible values
                    if key in search_space and isinstance(search_space[key], list):
                        individual[key] = np.random.choice(search_space[key])
    
    def _update_population(self, offspring: List[Dict[str, Any]]):
        """Update population with offspring."""
        # Combine parents and offspring
        combined = self.population + offspring

        # Sort by fitness
        combined_fitness = []
        for individual in combined:
            fitness = self._evaluate_individual(individual)
            combined_fitness.append(fitness)

        # Select best individuals
        sorted_indices = np.argsort(combined_fitness)[::-1]
        self.population = [combined[i] for i in sorted_indices[:self.config.population_size]]

    def _individual_to_weight_vector(self, individual: Dict[str, Any], n_features: int) -> np.ndarray:
        """Convert an individual configuration into a deterministic weight vector."""
        if n_features == 0:
            return np.array([], dtype=float)

        encoded_values: List[float] = []
        for key in sorted(individual.keys()):
            value = individual[key]
            if isinstance(value, bool):
                encoded_values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float, np.number)):
                encoded_values.append(float(value))
            elif isinstance(value, str):
                encoded_values.append(float(abs(hash((key, value))) % 1000) / 1000.0)
            else:
                encoded_values.append(float(abs(hash((key, str(value)))) % 1000) / 1000.0)

        if not encoded_values:
            encoded_values = [1.0]

        base_vector = np.array(encoded_values, dtype=float)
        base_vector = np.nan_to_num(base_vector, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(base_vector):
            base_vector = np.ones_like(base_vector)

        repeated = np.resize(base_vector, n_features)
        norm = float(np.linalg.norm(repeated))
        if norm == 0.0:
            repeated = np.ones(n_features, dtype=float) / np.sqrt(max(n_features, 1))
        else:
            repeated = repeated / norm
        return repeated

    def _derive_default_search_space(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create a lightweight default search space from market data."""
        numeric_columns = market_data.select_dtypes(include=[np.number])
        n_features = numeric_columns.shape[1]

        if n_features == 0:
            return {
                'scaling_factor': [0.5, 1.0, 1.5],
                'bias_term': [0.0, 0.1, 0.2],
            }

        feature_subset = min(max(n_features // 2, 1), n_features)
        feature_values = sorted({feature_subset, max(1, feature_subset - 1), min(n_features, feature_subset + 1)})
        return {
            'feature_subset': feature_values,
            'scaling_factor': [0.5, 1.0, 1.5],
            'regularization_strength': [0.0, 0.01, 0.05],
        }
