#!/usr/bin/env python3
"""
Multi-Objective HMM Regime Optimization Implementation

This implementation provides:
- NSGA-II algorithm for multi-objective optimization
- Four primary objectives: regime quality, efficiency, interpretability, robustness
- Pareto front analysis and visualization
- Interactive decision making
- Comprehensive reporting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import warnings
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')


@dataclass
class Individual:
    """Individual in the multi-objective optimization population."""
    params: Dict[str, Any]
    objectives: List[float] = field(default_factory=list)
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    domination_count: int = 0
    dominated_solutions: List['Individual'] = field(default_factory=list)
    strength: int = 0
    raw_fitness: float = 0.0
    density: float = 0.0


@dataclass
class MultiObjectiveMetrics:
    """Container for multi-objective optimization metrics."""
    regime_quality: float = 0.0
    computational_efficiency: float = 0.0
    interpretability: float = 0.0
    robustness: float = 0.0


class MultiObjectiveHMMOptimizer:
    """Multi-objective HMM regime optimizer using NSGA-II."""

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 random_state: int = 42):

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state

        self.population = []
        self.archive = []  # Pareto front archive
        self.optimization_history = []

        np.random.seed(random_state)

    def optimize(self,
                data: pd.DataFrame,
                feature_columns: List[str],
                market_condition_columns: List[str]) -> Dict[str, Any]:
        """Run multi-objective optimization."""

        print(f"🚀 Starting Multi-Objective HMM Optimization...")
        print(f"📊 Population size: {self.population_size}")
        print(f"🔄 Generations: {self.generations}")
        print(f"📈 Data shape: {data.shape}")

        # Initialize population
        self.population = self._initialize_population()

        # Main optimization loop
        for generation in range(self.generations):
            print(f"🔄 Generation {generation + 1}/{self.generations}")

            # Evaluate objectives for all individuals
            self._evaluate_population(data, feature_columns, market_condition_columns)

            # Non-dominated sorting
            fronts = self._non_dominated_sort()

            # Calculate crowding distance
            self._calculate_crowding_distance(fronts)

            # Selection
            parents = self._tournament_selection()

            # Crossover and mutation
            offspring = self._generate_offspring(parents)

            # Combine parent and offspring populations
            combined_population = self.population + offspring

            # Environmental selection
            self.population = self._environmental_selection(combined_population)

            # Update archive
            self._update_archive()

            # Log progress
            self._log_generation_progress(generation)

        # Final analysis
        final_results = self._final_analysis()

        print(f"✅ Multi-objective optimization completed!")
        print(f"🏆 Pareto front size: {len(self.archive)}")

        return final_results

    def _initialize_population(self) -> List[Individual]:
        """Initialize random population."""

        population = []

        for _ in range(self.population_size):
            params = self._generate_random_params()
            individual = Individual(params=params)
            population.append(individual)

        return population

    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameters for an individual."""

        return {
            'n_components': np.random.randint(2, 11),
            'covariance_type': np.random.choice(['full', 'tied', 'diag', 'spherical']),
            'n_iter': np.random.randint(100, 301),
            'tol': np.random.uniform(1e-4, 1e-2),
            'reg_covar': np.random.uniform(1e-6, 1e-3),
            'clustering_method': np.random.choice(['kmeans', 'gaussian_mixture']),
            'n_clusters': np.random.randint(3, 16),
            'target_regimes': np.random.randint(15, 21),
            'merging_method': np.random.choice(['hierarchical', 'kmeans', 'dbscan', 'spectral']),
            'similarity_threshold': np.random.uniform(0.3, 0.8),
            'coherence_threshold': np.random.uniform(0.6, 0.9),
            'differentiation_threshold': np.random.uniform(0.4, 0.8)
        }

    def _evaluate_population(self,
                           data: pd.DataFrame,
                           feature_columns: List[str],
                           market_condition_columns: List[str]):
        """Evaluate all objectives for the entire population."""

        for individual in self.population:
            try:
                # Generate clusters
                cluster_data = self._generate_clusters(individual.params, data)

                # Calculate all objectives
                objectives = self._calculate_all_objectives(
                    cluster_data, individual.params, data, market_condition_columns
                )

                individual.objectives = [
                    objectives.regime_quality,
                    objectives.computational_efficiency,
                    objectives.interpretability,
                    objectives.robustness
                ]

            except Exception as e:
                # Assign worst possible objectives for failed individuals
                individual.objectives = [0.0, 0.0, 0.0, 0.0]
                print(f"⚠️ Individual evaluation failed: {e}")

    def _generate_clusters(self, params: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
        """Generate clusters using given parameters."""

        # This is a simplified implementation
        # In practice, this would use the actual HMM and clustering logic

        result_data = data.copy()

        # Generate random clusters based on target_regimes
        target_regimes = params.get('target_regimes', 18)
        result_data['composite_cluster_id'] = np.random.randint(
            0, target_regimes, size=len(data)
        )

        return result_data

    def _calculate_all_objectives(self,
                                cluster_data: pd.DataFrame,
                                params: Dict[str, Any],
                                data: pd.DataFrame,
                                market_condition_columns: List[str]) -> MultiObjectiveMetrics:
        """Calculate all four objectives."""

        # 1. Regime Quality
        regime_quality = self._calculate_regime_quality_objective(
            cluster_data, market_condition_columns, params
        )

        # 2. Computational Efficiency
        computational_efficiency = self._calculate_efficiency_objective(params)

        # 3. Interpretability
        interpretability = self._calculate_interpretability_objective(cluster_data, params)

        # 4. Robustness
        robustness = self._calculate_robustness_objective(cluster_data, data, params)

        return MultiObjectiveMetrics(
            regime_quality=regime_quality,
            computational_efficiency=computational_efficiency,
            interpretability=interpretability,
            robustness=robustness
        )

    def _calculate_regime_quality_objective(self,
                                          cluster_data: pd.DataFrame,
                                          market_condition_columns: List[str],
                                          params: Dict[str, Any]) -> float:
        """Calculate regime quality objective."""

        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0

        # Calculate individual metrics
        differentiation = self._calculate_regime_differentiation(cluster_data, market_condition_columns)
        coherence = self._calculate_internal_coherence(cluster_data, market_condition_columns)
        persistence = self._calculate_regime_persistence(cluster_data)
        smoothness = self._calculate_transition_smoothness(cluster_data)

        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]
        metrics = [differentiation, coherence, persistence, smoothness]

        regime_quality = np.average(metrics, weights=weights)

        return regime_quality

    def _calculate_regime_differentiation(self, cluster_data: pd.DataFrame,
                                        market_condition_columns: List[str]) -> float:
        """Calculate regime differentiation."""

        if not market_condition_columns or 'composite_cluster_id' not in cluster_data.columns:
            return 0.0

        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]
        if not valid_columns:
            return 0.0

        differentiation_scores = []

        for col in valid_columns:
            regime_means = cluster_data.groupby('composite_cluster_id')[col].mean()

            if len(regime_means) < 2:
                continue

            # Calculate pairwise differences
            means_array = regime_means.values
            n_regimes = len(means_array)

            differences = []
            for i in range(n_regimes):
                for j in range(i + 1, n_regimes):
                    differences.append(abs(means_array[i] - means_array[j]))

            if differences:
                # Normalize by overall range
                overall_range = cluster_data[col].max() - cluster_data[col].min()
                if overall_range > 0:
                    avg_difference = np.mean(differences) / overall_range
                    differentiation_scores.append(avg_difference)

        return np.mean(differentiation_scores) if differentiation_scores else 0.0

    def _calculate_internal_coherence(self, cluster_data: pd.DataFrame,
                                    market_condition_columns: List[str]) -> float:
        """Calculate internal coherence."""

        if not market_condition_columns or 'composite_cluster_id' not in cluster_data.columns:
            return 0.0

        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]
        if not valid_columns:
            return 0.0

        coherence_scores = []

        for col in valid_columns:
            regime_stats = cluster_data.groupby('composite_cluster_id')[col].agg(['mean', 'std', 'count'])
            valid_regimes = regime_stats[regime_stats['count'] > 1]

            if len(valid_regimes) > 0:
                means = valid_regimes['mean'].values
                stds = valid_regimes['std'].values

                # Calculate coefficient of variation
                non_zero_means = means != 0
                if np.any(non_zero_means):
                    cvs = stds[non_zero_means] / np.abs(means[non_zero_means])

                    if len(cvs) > 0:
                        avg_cv = np.mean(cvs)
                        coherence = 1.0 / (1.0 + avg_cv)
                        coherence_scores.append(coherence)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _calculate_regime_persistence(self, cluster_data: pd.DataFrame) -> float:
        """Calculate regime persistence."""

        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0

        cluster_series = cluster_data['composite_cluster_id'].values

        # Calculate regime changes
        regime_changes = np.diff(cluster_series) != 0
        total_periods = len(cluster_series)

        if total_periods == 0:
            return 0.0

        # Calculate persistence as average regime duration
        change_indices = np.where(regime_changes)[0]

        if len(change_indices) == 0:
            # No changes - perfect persistence
            return 1.0

        # Calculate regime durations
        durations = []
        prev_change = -1

        for change_idx in change_indices:
            duration = change_idx - prev_change
            durations.append(duration)
            prev_change = change_idx

        # Add final regime duration
        final_duration = total_periods - prev_change - 1
        durations.append(final_duration)

        # Calculate average persistence
        avg_duration = np.mean(durations)
        max_possible_duration = total_periods

        persistence = avg_duration / max_possible_duration

        return min(1.0, persistence)

    def _calculate_transition_smoothness(self, cluster_data: pd.DataFrame) -> float:
        """Calculate transition smoothness."""

        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0

        cluster_series = cluster_data['composite_cluster_id'].values

        # Calculate transition probabilities
        unique_regimes = np.unique(cluster_series)
        n_regimes = len(unique_regimes)

        if n_regimes < 2:
            return 0.0

        # Create transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}

        for i in range(len(cluster_series) - 1):
            current_regime = cluster_series[i]
            next_regime = cluster_series[i + 1]

            current_idx = regime_to_idx[current_regime]
            next_idx = regime_to_idx[next_regime]

            transition_matrix[current_idx, next_idx] += 1

        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]

        # Calculate smoothness as entropy of transition probabilities
        # Lower entropy = smoother transitions
        entropy = 0.0
        for row in transition_matrix:
            row = row[row > 0]  # Remove zero probabilities
            if len(row) > 0:
                entropy += -np.sum(row * np.log(row))

        # Normalize by maximum possible entropy
        max_entropy = n_regimes * np.log(n_regimes)
        if max_entropy > 0:
            smoothness = 1.0 - (entropy / max_entropy)
        else:
            smoothness = 0.0

        return max(0.0, smoothness)

    def _calculate_efficiency_objective(self, params: Dict[str, Any]) -> float:
        """Calculate computational efficiency objective."""

        # Parameter complexity penalty
        complexity_penalty = self._calculate_parameter_complexity(params)

        # Simple efficiency score (in practice, would measure actual time/memory)
        efficiency = 1.0 - complexity_penalty

        return max(0.0, efficiency)

    def _calculate_parameter_complexity(self, params: Dict[str, Any]) -> float:
        """Calculate parameter complexity penalty."""

        complexity = 0

        # HMM complexity
        complexity += params.get('n_components', 5) / 10  # Normalize by max components

        # Clustering complexity
        complexity += params.get('n_clusters', 10) / 20  # Normalize by max clusters

        # Iteration complexity
        max_iter = params.get('n_iter', 100)
        complexity += min(max_iter / 500, 0.2)

        return min(complexity, 1.0)

    def _calculate_interpretability_objective(self, cluster_data: pd.DataFrame,
                                            params: Dict[str, Any]) -> float:
        """Calculate interpretability objective."""

        interpretability = 0

        # Regime count penalty (prefer 15-20 regimes)
        n_regimes = len(cluster_data['composite_cluster_id'].unique())
        target_regimes = params.get('target_regimes', 18)

        if 15 <= n_regimes <= 20:
            regime_count_score = 1.0
        else:
            penalty = abs(n_regimes - target_regimes) / target_regimes
            regime_count_score = max(0, 1 - penalty)

        # Regime balance score
        regime_sizes = cluster_data['composite_cluster_id'].value_counts()
        balance_score = 1.0 / (1.0 + regime_sizes.std() / regime_sizes.mean())

        # Parameter simplicity score
        simplicity_score = self._calculate_parameter_simplicity(params)

        # Weighted combination
        weights = [0.4, 0.3, 0.3]
        scores = [regime_count_score, balance_score, simplicity_score]

        interpretability = np.average(scores, weights=weights)

        return interpretability

    def _calculate_parameter_simplicity(self, params: Dict[str, Any]) -> float:
        """Calculate parameter simplicity score."""

        simplicity = 1.0

        # Penalize complex covariance types
        if params.get('covariance_type') == 'full':
            simplicity -= 0.2

        # Penalize complex merging methods
        if params.get('merging_method') in ['spectral', 'dbscan']:
            simplicity -= 0.1

        # Penalize high iteration counts
        max_iter = params.get('n_iter', 100)
        simplicity -= min(max_iter / 500, 0.2)

        return max(simplicity, 0.0)

    def _calculate_robustness_objective(self, cluster_data: pd.DataFrame,
                                      data: pd.DataFrame,
                                      params: Dict[str, Any]) -> float:
        """Calculate robustness objective."""

        # Simplified robustness calculation
        # In practice, would include cross-validation, bootstrap, etc.

        robustness = 0.5  # Base score

        # Add some randomness to simulate robustness variation
        robustness += np.random.normal(0, 0.1)

        return max(0.0, min(1.0, robustness))

    def _non_dominated_sort(self) -> List[List[Individual]]:
        """Perform non-dominated sorting."""

        fronts = [[]]

        for individual in self.population:
            individual.domination_count = 0
            individual.dominated_solutions = []

            for other in self.population:
                if self._dominates(individual, other):
                    individual.dominated_solutions.append(other)
                elif self._dominates(other, individual):
                    individual.domination_count += 1

            if individual.domination_count == 0:
                individual.pareto_rank = 0
                fronts[0].append(individual)

        i = 0
        while fronts[i]:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.pareto_rank = i + 1
                        next_front.append(dominated)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, individual1: Individual, individual2: Individual) -> bool:
        """Check if individual1 dominates individual2."""

        objectives1 = individual1.objectives
        objectives2 = individual2.objectives

        # Check if individual1 is at least as good in all objectives
        at_least_as_good = all(obj1 >= obj2 for obj1, obj2 in zip(objectives1, objectives2))

        # Check if individual1 is strictly better in at least one objective
        strictly_better = any(obj1 > obj2 for obj1, obj2 in zip(objectives1, objectives2))

        return at_least_as_good and strictly_better

    def _calculate_crowding_distance(self, fronts: List[List[Individual]]):
        """Calculate crowding distance for all individuals."""

        for front in fronts:
            if len(front) <= 2:
                # Assign infinite crowding distance to boundary solutions
                for individual in front:
                    individual.crowding_distance = float('inf')
                continue

            # Calculate crowding distance for each objective
            n_objectives = len(front[0].objectives)

            for obj_idx in range(n_objectives):
                # Sort front by objective
                front.sort(key=lambda x: x.objectives[obj_idx])

                # Set boundary solutions to infinite distance
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')

                # Calculate crowding distance for intermediate solutions
                obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]

                if obj_range == 0:
                    continue

                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] -
                              front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance

    def _tournament_selection(self) -> List[Individual]:
        """Perform tournament selection."""

        parents = []

        while len(parents) < self.population_size:
            # Select two random individuals
            idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
            individual1 = self.population[idx1]
            individual2 = self.population[idx2]

            # Select winner based on Pareto rank and crowding distance
            if individual1.pareto_rank < individual2.pareto_rank:
                winner = individual1
            elif individual1.pareto_rank > individual2.pareto_rank:
                winner = individual2
            else:
                # Same rank, use crowding distance
                if individual1.crowding_distance > individual2.crowding_distance:
                    winner = individual1
                else:
                    winner = individual2

            parents.append(winner)

        return parents

    def _generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Generate offspring through crossover and mutation."""

        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                else:
                    child1, child2 = parents[i], parents[i + 1]

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)

                offspring.extend([child1, child2])
            else:
                # Single parent
                child = parents[i]
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child)
                offspring.append(child)

        return offspring

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""

        # Uniform crossover
        child1_params = {}
        child2_params = {}

        for key in parent1.params:
            if np.random.random() < 0.5:
                child1_params[key] = parent1.params[key]
                child2_params[key] = parent2.params[key]
            else:
                child1_params[key] = parent2.params[key]
                child2_params[key] = parent1.params[key]

        child1 = Individual(params=child1_params)
        child2 = Individual(params=child2_params)

        return child1, child2

    def _mutate(self, individual: Individual) -> Individual:
        """Perform mutation on an individual."""

        mutated_params = individual.params.copy()

        # Randomly mutate some parameters
        for key in mutated_params:
            if np.random.random() < 0.1:  # 10% mutation probability per parameter
                if key == 'n_components':
                    mutated_params[key] = np.random.randint(2, 11)
                elif key == 'covariance_type':
                    mutated_params[key] = np.random.choice(['full', 'tied', 'diag', 'spherical'])
                elif key == 'n_iter':
                    mutated_params[key] = np.random.randint(100, 301)
                elif key == 'tol':
                    mutated_params[key] = np.random.uniform(1e-4, 1e-2)
                elif key == 'reg_covar':
                    mutated_params[key] = np.random.uniform(1e-6, 1e-3)
                elif key == 'clustering_method':
                    mutated_params[key] = np.random.choice(['kmeans', 'gaussian_mixture'])
                elif key == 'n_clusters':
                    mutated_params[key] = np.random.randint(3, 16)
                elif key == 'target_regimes':
                    mutated_params[key] = np.random.randint(15, 21)
                elif key == 'merging_method':
                    mutated_params[key] = np.random.choice(['hierarchical', 'kmeans', 'dbscan', 'spectral'])
                elif key == 'similarity_threshold':
                    mutated_params[key] = np.random.uniform(0.3, 0.8)
                elif key == 'coherence_threshold':
                    mutated_params[key] = np.random.uniform(0.6, 0.9)
                elif key == 'differentiation_threshold':
                    mutated_params[key] = np.random.uniform(0.4, 0.8)

        return Individual(params=mutated_params)

    def _environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """Perform environmental selection."""

        # Non-dominated sorting of combined population
        fronts = self._non_dominated_sort_combined(combined_population)

        # Calculate crowding distance
        self._calculate_crowding_distance_combined(fronts)

        # Select individuals for next generation
        new_population = []

        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Sort front by crowding distance and fill remaining slots
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining_slots = self.population_size - len(new_population)
                new_population.extend(front[:remaining_slots])
                break

        return new_population

    def _non_dominated_sort_combined(self, population: List[Individual]) -> List[List[Individual]]:
        """Non-dominated sorting for combined population."""

        fronts = [[]]

        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []

            for other in population:
                if self._dominates(individual, other):
                    individual.dominated_solutions.append(other)
                elif self._dominates(other, individual):
                    individual.domination_count += 1

            if individual.domination_count == 0:
                individual.pareto_rank = 0
                fronts[0].append(individual)

        i = 0
        while fronts[i]:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.pareto_rank = i + 1
                        next_front.append(dominated)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _calculate_crowding_distance_combined(self, fronts: List[List[Individual]]):
        """Calculate crowding distance for combined population."""

        for front in fronts:
            if len(front) <= 2:
                for individual in front:
                    individual.crowding_distance = float('inf')
                continue

            n_objectives = len(front[0].objectives)

            for obj_idx in range(n_objectives):
                front.sort(key=lambda x: x.objectives[obj_idx])

                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')

                obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]

                if obj_range == 0:
                    continue

                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] -
                              front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance

    def _update_archive(self):
        """Update Pareto front archive."""

        # Find non-dominated solutions in current population
        non_dominated = []

        for individual in self.population:
            is_dominated = False
            for other in self.population:
                if self._dominates(other, individual):
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.append(individual)

        # Add to archive
        self.archive.extend(non_dominated)

        # Remove dominated solutions from archive
        self.archive = self._remove_dominated_from_archive()

    def _remove_dominated_from_archive(self) -> List[Individual]:
        """Remove dominated solutions from archive."""

        non_dominated = []

        for individual in self.archive:
            is_dominated = False
            for other in self.archive:
                if individual != other and self._dominates(other, individual):
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.append(individual)

        return non_dominated

    def _log_generation_progress(self, generation: int):
        """Log progress for current generation."""

        # Calculate statistics
        objectives_matrix = np.array([ind.objectives for ind in self.population])

        stats = {
            'generation': generation,
            'population_size': len(self.population),
            'archive_size': len(self.archive),
            'mean_regime_quality': np.mean(objectives_matrix[:, 0]),
            'mean_efficiency': np.mean(objectives_matrix[:, 1]),
            'mean_interpretability': np.mean(objectives_matrix[:, 2]),
            'mean_robustness': np.mean(objectives_matrix[:, 3])
        }

        self.optimization_history.append(stats)

        print(f"   📊 Archive size: {len(self.archive)}")
        print(f"   🎯 Mean objectives: Q={stats['mean_regime_quality']:.3f}, "
              f"E={stats['mean_efficiency']:.3f}, I={stats['mean_interpretability']:.3f}, "
              f"R={stats['mean_robustness']:.3f}")

    def _final_analysis(self) -> Dict[str, Any]:
        """Perform final analysis of optimization results."""

        # Remove duplicates from archive
        unique_archive = self._remove_duplicates(self.archive)

        # Calculate Pareto front metrics
        pareto_metrics = self._calculate_pareto_metrics(unique_archive)

        # Find knee points
        knee_points = self._find_knee_points(unique_archive)

        return {
            'archive': unique_archive,
            'pareto_metrics': pareto_metrics,
            'knee_points': knee_points,
            'optimization_history': self.optimization_history
        }

    def _remove_duplicates(self, archive: List[Individual]) -> List[Individual]:
        """Remove duplicate solutions from archive."""

        unique_archive = []
        seen_objectives = set()

        for individual in archive:
            objectives_tuple = tuple(individual.objectives)
            if objectives_tuple not in seen_objectives:
                unique_archive.append(individual)
                seen_objectives.add(objectives_tuple)

        return unique_archive

    def _calculate_pareto_metrics(self, archive: List[Individual]) -> Dict[str, float]:
        """Calculate Pareto front metrics."""

        if not archive:
            return {}

        objectives_matrix = np.array([ind.objectives for ind in archive])

        # Calculate hypervolume (simplified)
        hypervolume = self._calculate_hypervolume(objectives_matrix)

        # Calculate spread
        spread = self._calculate_spread(objectives_matrix)

        # Calculate uniformity
        uniformity = self._calculate_uniformity(objectives_matrix)

        return {
            'hypervolume': hypervolume,
            'spread': spread,
            'uniformity': uniformity,
            'size': len(archive)
        }

    def _calculate_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate hypervolume indicator (simplified)."""

        # Simplified hypervolume calculation
        # In practice, would use proper hypervolume calculation

        # Use product of objective ranges as approximation
        ranges = np.max(objectives_matrix, axis=0) - np.min(objectives_matrix, axis=0)
        hypervolume = np.prod(ranges)

        return hypervolume

    def _calculate_spread(self, objectives_matrix: np.ndarray) -> float:
        """Calculate spread of Pareto front."""

        # Calculate maximum distance between any two points
        max_distance = 0

        for i in range(len(objectives_matrix)):
            for j in range(i + 1, len(objectives_matrix)):
                distance = np.linalg.norm(objectives_matrix[i] - objectives_matrix[j])
                max_distance = max(max_distance, distance)

        return max_distance

    def _calculate_uniformity(self, objectives_matrix: np.ndarray) -> float:
        """Calculate uniformity of Pareto front."""

        if len(objectives_matrix) < 2:
            return 1.0

        # Calculate average distance between consecutive points
        distances = []

        for i in range(len(objectives_matrix) - 1):
            distance = np.linalg.norm(objectives_matrix[i] - objectives_matrix[i + 1])
            distances.append(distance)

        if not distances:
            return 1.0

        # Uniformity is inverse of standard deviation
        uniformity = 1.0 / (1.0 + np.std(distances))

        return uniformity

    def _find_knee_points(self, archive: List[Individual]) -> List[Individual]:
        """Find knee points in Pareto front."""

        if len(archive) < 3:
            return archive

        knee_points = []

        for i, individual in enumerate(archive):
            # Calculate angle with neighbors
            angles = []

            for j, other in enumerate(archive):
                if i != j:
                    angle = self._calculate_angle(individual, other)
                    angles.append(angle)

            # If angle is significantly different from neighbors, it's a knee point
            if angles:
                mean_angle = np.mean(angles)
                std_angle = np.std(angles)

                if abs(angles[0] - mean_angle) > 2 * std_angle:
                    knee_points.append(individual)

        return knee_points

    def _calculate_angle(self, individual1: Individual, individual2: Individual) -> float:
        """Calculate angle between two individuals in objective space."""

        obj1 = np.array(individual1.objectives)
        obj2 = np.array(individual2.objectives)

        # Calculate angle between vectors
        dot_product = np.dot(obj1, obj2)
        norm1 = np.linalg.norm(obj1)
        norm2 = np.linalg.norm(obj2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle)

    def visualize_pareto_front(self, archive: List[Individual] = None):
        """Visualize Pareto front."""

        if archive is None:
            archive = self.archive

        if not archive:
            print("No solutions in archive to visualize")
            return

        # Create 2D scatter plot
        objectives_matrix = np.array([ind.objectives for ind in archive])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Multi-Objective Optimization Results', fontsize=16)

        # Plot pairwise objective combinations
        objective_names = ['Regime Quality', 'Efficiency', 'Interpretability', 'Robustness']

        plot_idx = 0
        for i in range(4):
            for j in range(i + 1, 4):
                row = plot_idx // 3
                col = plot_idx % 3

                axes[row, col].scatter(objectives_matrix[:, i], objectives_matrix[:, j],
                                     alpha=0.6, s=50)
                axes[row, col].set_xlabel(objective_names[i])
                axes[row, col].set_ylabel(objective_names[j])
                axes[row, col].grid(True, alpha=0.3)

                plot_idx += 1

        plt.tight_layout()
        plt.show()

    def interactive_selection(self, archive: List[Individual] = None):
        """Interactive Pareto front selection."""

        if archive is None:
            archive = self.archive

        if not archive:
            print("No solutions in archive for selection")
            return []

        print("Interactive Pareto Front Selection")
        print("=" * 60)

        selected_solutions = []

        while True:
            # Display current solutions
            self._display_solutions(archive)

            # Get user preference
            choice = input("\nEnter solution number to select (or 'q' to quit): ")

            if choice.lower() == 'q':
                break

            try:
                solution_index = int(choice)
                if 0 <= solution_index < len(archive):
                    selected_solution = archive[solution_index]
                    selected_solutions.append(selected_solution)
                    print(f"✅ Selected solution {solution_index}")
                    print(f"   Parameters: {selected_solution.params}")
                    print(f"   Objectives: {selected_solution.objectives}")
                else:
                    print("❌ Invalid solution number")
            except ValueError:
                print("❌ Invalid input")

        return selected_solutions

    def _display_solutions(self, archive: List[Individual]):
        """Display available solutions."""

        print("\nAvailable Solutions:")
        print("-" * 100)
        print(f"{'Index':<6} {'Regime Quality':<15} {'Efficiency':<12} {'Interpretability':<15} {'Robustness':<12}")
        print("-" * 100)

        for i, individual in enumerate(archive):
            objectives = individual.objectives
            print(f"{i:<6} {objectives[0]:<15.4f} {objectives[1]:<12.4f} {objectives[2]:<15.4f} {objectives[3]:<12.4f}")


def main():
    """Example usage of multi-objective optimizer."""

    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    n_features = 10

    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add market condition columns
    data['volatility'] = np.random.exponential(1, n_samples)
    data['momentum'] = np.random.normal(0, 1, n_samples)
    data['volume'] = np.random.lognormal(0, 1, n_samples)
    data['returns'] = np.random.normal(0, 0.02, n_samples)

    feature_columns = [f'feature_{i}' for i in range(n_features)]
    market_condition_columns = ['volatility', 'momentum', 'volume', 'returns']

    # Initialize multi-objective optimizer
    optimizer = MultiObjectiveHMMOptimizer(
        population_size=50,  # Smaller for demo
        generations=20,      # Fewer for demo
        crossover_rate=0.8,
        mutation_rate=0.1
    )

    # Run optimization
    results = optimizer.optimize(data, feature_columns, market_condition_columns)

    # Display results
    print(f"\n🎉 Multi-objective optimization completed!")
    print(f"🏆 Pareto front size: {len(results['archive'])}")
    print(f"📊 Pareto metrics: {results['pareto_metrics']}")
    print(f"🎯 Knee points: {len(results['knee_points'])}")

    # Visualize results
    optimizer.visualize_pareto_front()

    # Interactive selection
    selected = optimizer.interactive_selection()
    print(f"\n✅ Selected {len(selected)} solutions")


if __name__ == "__main__":
    main()