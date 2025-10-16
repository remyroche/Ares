"""
Multi-Objective Optimizer for Hybrid NAS-TAS Regime Discovery.

Implements advanced multi-objective optimization with CV minimization,
cluster distribution targets, and Pareto frontier analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import itertools

logger = logging.getLogger(__name__)

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    # Optimization targets
    target_cluster_count_min: int = 6
    target_cluster_count_max: int = 15
    max_cluster_distribution: float = 0.25  # 25% max
    min_cluster_distribution: float = 0.03  # 3% min

    # CV optimization weights
    volatility_cv_weight: float = 0.4
    returns_cv_weight: float = 0.3
    volume_cv_weight: float = 0.3

    # Removed accessory CV weights (momentum and entropy)

    # Multi-objective weights
    statistical_weight: float = 0.25
    economic_weight: float = 0.30
    temporal_weight: float = 0.20
    cv_optimization_weight: float = 0.25

    # Optimization parameters
    max_iterations: int = 100
    population_size: int = 50
    convergence_threshold: float = 0.01
    enable_pareto_frontier: bool = True
    pareto_frontier_size: int = 20

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for regime clustering with CV optimization.

    Optimizes multiple objectives simultaneously:
    1. Statistical clustering quality
    2. Economic significance
    3. Temporal consistency
    4. CV optimization (volatility, returns, volume)
    5. Cluster distribution targets
    """

    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        """Initialize the multi-objective optimizer."""
        self.config = config or MultiObjectiveConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_regime_clustering(self, nas_predictions: np.ndarray,
                                 tas_predictions: np.ndarray,
                                 market_data: pd.DataFrame,
                                 features: Optional[pd.DataFrame] = None,
                                 initial_consensus: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize regime clustering using multi-objective optimization.

        Args:
            nas_predictions: NAS regime predictions
            tas_predictions: TAS regime predictions
            market_data: Market data for optimization
            features: Optional features for optimization
            initial_consensus: Optional initial consensus predictions

        Returns:
            Optimization results with Pareto frontier
        """
        try:
            self.logger.info("🚀 Starting multi-objective regime clustering optimization")

            # Prepare data
            prepared_data = self._prepare_optimization_data(
                nas_predictions, tas_predictions, market_data, features
            )

            # Generate initial solutions
            initial_solutions = self._generate_initial_solutions(
                prepared_data, initial_consensus
            )

            # Perform multi-objective optimization
            if self.config.enable_pareto_frontier:
                optimization_result = self._pareto_optimization(
                    initial_solutions, prepared_data
                )
            else:
                optimization_result = self._single_objective_optimization(
                    initial_solutions, prepared_data
                )

            # Evaluate final solutions
            final_evaluation = self._evaluate_optimization_results(
                optimization_result, prepared_data
            )

            # Select best solution
            best_solution = self._select_best_solution(
                optimization_result, final_evaluation
            )

            results = {
                'optimization_result': optimization_result,
                'final_evaluation': final_evaluation,
                'best_solution': best_solution,
                'pareto_frontier': optimization_result.get('pareto_frontier', []),
                'optimization_success': True
            }

            self.logger.info("✅ Multi-objective optimization completed")
            return results

        except Exception as e:
            self.logger.error(f"❌ Multi-objective optimization failed: {e}")
            return {'error': str(e), 'optimization_success': False}

    def _prepare_optimization_data(self, nas_predictions: np.ndarray,
                                 tas_predictions: np.ndarray,
                                 market_data: pd.DataFrame,
                                 features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Prepare data for optimization."""
        try:
            # Align predictions
            min_length = min(len(nas_predictions), len(tas_predictions))
            nas_predictions = nas_predictions[:min_length]
            tas_predictions = tas_predictions[:min_length]
            market_data = market_data.iloc[:min_length]

            # Prepare features
            if features is not None and not features.empty:
                optimization_features = features.iloc[:min_length].values
            else:
                # Use market data features
                numeric_columns = market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    optimization_features = market_data[numeric_columns].values
                else:
                    # Fallback to basic OHLCV
                    basic_columns = ['open', 'high', 'low', 'close', 'volume']
                    available_columns = [col for col in basic_columns if col in market_data.columns]
                    optimization_features = market_data[available_columns].values if available_columns else market_data.values

            # Calculate market characteristics
            market_characteristics = self._calculate_market_characteristics(market_data)

            return {
                'nas_predictions': nas_predictions,
                'tas_predictions': tas_predictions,
                'market_data': market_data,
                'features': optimization_features,
                'market_characteristics': market_characteristics,
                'data_length': min_length
            }

        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            return {'error': str(e)}

    def _calculate_market_characteristics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market characteristics for optimization."""
        try:
            characteristics = {}

            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                characteristics['returns'] = returns.values
                characteristics['volatility'] = returns.std()
                characteristics['mean_return'] = returns.mean()
                characteristics['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                characteristics['returns'] = np.zeros(len(market_data))
                characteristics['volatility'] = 0.0
                characteristics['mean_return'] = 0.0
                characteristics['sharpe_ratio'] = 0.0

            if 'volume' in market_data.columns:
                volume = market_data['volume']
                characteristics['volume'] = volume.values
                characteristics['volume_mean'] = volume.mean()
                characteristics['volume_std'] = volume.std()
            else:
                characteristics['volume'] = np.ones(len(market_data))
                characteristics['volume_mean'] = 1.0
                characteristics['volume_std'] = 0.0

            # Removed momentum and entropy calculations

            return characteristics

        except Exception as e:
            self.logger.error(f"❌ Market characteristics calculation failed: {e}")
            return {}

    def _calculate_momentum(self, returns: np.ndarray) -> np.ndarray:
        """Calculate momentum from returns."""
        try:
            momentum = np.zeros_like(returns)
            for i in range(5, len(returns)):  # 5-period momentum
                momentum[i] = np.sum(returns[i-5:i])
            return momentum
        except Exception:
            return np.zeros_like(returns)

    def _calculate_entropy(self, returns: np.ndarray) -> np.ndarray:
        """Calculate entropy from returns."""
        try:
            # Discretize returns into bins
            n_bins = 10
            bins = np.digitize(returns, np.linspace(returns.min(), returns.max(), n_bins))

            # Calculate entropy for rolling window
            window_size = 20
            entropy_values = np.zeros_like(returns)

            for i in range(window_size, len(returns)):
                window_bins = bins[i-window_size:i]
                bin_counts = np.bincount(window_bins, minlength=n_bins)
                probabilities = bin_counts / len(window_bins)
                probabilities = probabilities[probabilities > 0]
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropy_values[i] = entropy

            return entropy_values
        except Exception:
            return np.zeros_like(returns)

    def _generate_initial_solutions(self, prepared_data: Dict[str, Any],
                                  initial_consensus: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Generate initial solutions for optimization."""
        try:
            nas_predictions = prepared_data['nas_predictions']
            tas_predictions = prepared_data['tas_predictions']
            data_length = prepared_data['data_length']

            solutions = []

            # Solution 1: Simple consensus (majority vote)
            consensus_solution = self._create_consensus_solution(nas_predictions, tas_predictions)
            solutions.append(consensus_solution)

            # Solution 2: Weighted consensus
            weighted_solution = self._create_weighted_solution(nas_predictions, tas_predictions)
            solutions.append(weighted_solution)

            # Solution 3: Economic-weighted consensus
            economic_solution = self._create_economic_weighted_solution(
                nas_predictions, tas_predictions, prepared_data
            )
            solutions.append(economic_solution)

            # Solution 4: Random initialization
            random_solution = self._create_random_solution(data_length)
            solutions.append(random_solution)

            # Add initial consensus if provided
            if initial_consensus is not None:
                solutions.append(initial_consensus[:data_length])

            # Generate additional random solutions
            for _ in range(self.config.population_size - len(solutions)):
                random_solution = self._create_random_solution(data_length)
                solutions.append(random_solution)

            return solutions

        except Exception as e:
            self.logger.error(f"❌ Initial solution generation failed: {e}")
            return []

    def _create_consensus_solution(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray) -> np.ndarray:
        """Create simple consensus solution."""
        try:
            consensus = np.zeros_like(nas_predictions)
            for i in range(len(nas_predictions)):
                if nas_predictions[i] == tas_predictions[i]:
                    consensus[i] = nas_predictions[i]
                else:
                    consensus[i] = (nas_predictions[i] + tas_predictions[i]) % 10
            return consensus
        except Exception:
            return nas_predictions

    def _create_weighted_solution(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray) -> np.ndarray:
        """Create weighted consensus solution."""
        try:
            # Calculate weights based on regime frequency
            nas_weights = self._calculate_regime_weights(nas_predictions)
            tas_weights = self._calculate_regime_weights(tas_predictions)

            weighted_solution = np.zeros_like(nas_predictions)
            for i in range(len(nas_predictions)):
                nas_weight = nas_weights.get(nas_predictions[i], 0.5)
                tas_weight = tas_weights.get(tas_predictions[i], 0.5)

                if nas_weight > tas_weight:
                    weighted_solution[i] = nas_predictions[i]
                else:
                    weighted_solution[i] = tas_predictions[i]

            return weighted_solution
        except Exception:
            return nas_predictions

    def _create_economic_weighted_solution(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                                         prepared_data: Dict[str, Any]) -> np.ndarray:
        """Create economically weighted consensus solution."""
        try:
            market_characteristics = prepared_data['market_characteristics']
            returns = market_characteristics.get('returns', np.zeros(len(nas_predictions)))
            volume = market_characteristics.get('volume', np.ones(len(nas_predictions)))

            economic_solution = np.zeros_like(nas_predictions)
            for i in range(len(nas_predictions)):
                # Calculate economic weights
                nas_economic_weight = self._calculate_economic_weight(
                    nas_predictions[i], returns[i], volume[i]
                )
                tas_economic_weight = self._calculate_economic_weight(
                    tas_predictions[i], returns[i], volume[i]
                )

                if nas_economic_weight > tas_economic_weight:
                    economic_solution[i] = nas_predictions[i]
                else:
                    economic_solution[i] = tas_predictions[i]

            return economic_solution
        except Exception:
            return nas_predictions

    def _create_random_solution(self, data_length: int) -> np.ndarray:
        """Create random solution."""
        try:
            n_regimes = np.random.randint(self.config.target_cluster_count_min,
                                        self.config.target_cluster_count_max + 1)
            return np.random.randint(0, n_regimes, data_length)
        except Exception:
            return np.zeros(data_length)

    def _calculate_regime_weights(self, predictions: np.ndarray) -> Dict[int, float]:
        """Calculate weights for each regime based on frequency."""
        try:
            unique_regimes, counts = np.unique(predictions, return_counts=True)
            total_count = len(predictions)
            weights = {regime: count / total_count for regime, count in zip(unique_regimes, counts)}
            return weights
        except Exception:
            return {}

    def _calculate_economic_weight(self, regime: int, return_val: float, volume_val: float) -> float:
        """Calculate economic weight for a regime."""
        try:
            # Simple economic weight based on return and volume
            return_weight = abs(return_val) * 10  # Scale return
            volume_weight = min(volume_val / 1000, 1.0)  # Normalize volume
            return return_weight + volume_weight
        except Exception:
            return 0.5

    def _pareto_optimization(self, initial_solutions: List[np.ndarray],
                           prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Pareto frontier optimization."""
        try:
            self.logger.info("🔍 Starting Pareto frontier optimization")

            # Evaluate initial solutions
            evaluated_solutions = []
            for solution in initial_solutions:
                objectives = self._evaluate_solution_objectives(solution, prepared_data)
                evaluated_solutions.append({
                    'solution': solution,
                    'objectives': objectives
                })

            # Find Pareto frontier
            pareto_frontier = self._find_pareto_frontier(evaluated_solutions)

            # Evolve solutions
            for iteration in range(self.config.max_iterations):
                # Generate new solutions through crossover and mutation
                new_solutions = self._evolve_solutions(pareto_frontier, prepared_data)

                # Evaluate new solutions
                for solution in new_solutions:
                    objectives = self._evaluate_solution_objectives(solution, prepared_data)
                    evaluated_solutions.append({
                        'solution': solution,
                        'objectives': objectives
                    })

                # Update Pareto frontier
                pareto_frontier = self._find_pareto_frontier(evaluated_solutions)

                # Check convergence
                if self._check_convergence(pareto_frontier, iteration):
                    break

            return {
                'pareto_frontier': pareto_frontier,
                'all_solutions': evaluated_solutions,
                'optimization_iterations': iteration + 1
            }

        except Exception as e:
            self.logger.error(f"❌ Pareto optimization failed: {e}")
            return {'error': str(e), 'pareto_frontier': []}

    def _single_objective_optimization(self, initial_solutions: List[np.ndarray],
                                     prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform single-objective optimization."""
        try:
            self.logger.info("🎯 Starting single-objective optimization")

            best_solution = None
            best_score = float('-inf')

            for solution in initial_solutions:
                objectives = self._evaluate_solution_objectives(solution, prepared_data)
                combined_score = self._combine_objectives(objectives)

                if combined_score > best_score:
                    best_score = combined_score
                    best_solution = solution

            return {
                'best_solution': best_solution,
                'best_score': best_score,
                'optimization_type': 'single_objective'
            }

        except Exception as e:
            self.logger.error(f"❌ Single-objective optimization failed: {e}")
            return {'error': str(e)}

    def _evaluate_solution_objectives(self, solution: np.ndarray,
                                    prepared_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate multiple objectives for a solution."""
        try:
            objectives = {}

            # Statistical objectives
            objectives['silhouette_score'] = self._calculate_silhouette_score(solution, prepared_data)
            objectives['calinski_harabasz_score'] = self._calculate_calinski_harabasz_score(solution, prepared_data)
            objectives['davies_bouldin_score'] = self._calculate_davies_bouldin_score(solution, prepared_data)

            # Economic objectives
            objectives['economic_significance'] = self._calculate_economic_significance(solution, prepared_data)
            objectives['trading_viability'] = self._calculate_trading_viability(solution, prepared_data)

            # CV optimization objectives
            objectives['cv_optimization'] = self._calculate_cv_optimization(solution, prepared_data)

            # Distribution objectives
            objectives['distribution_quality'] = self._calculate_distribution_quality(solution)

            # Temporal objectives
            objectives['temporal_consistency'] = self._calculate_temporal_consistency(solution)

            return objectives

        except Exception as e:
            self.logger.error(f"❌ Solution objectives evaluation failed: {e}")
            return {}

    def _calculate_silhouette_score(self, solution: np.ndarray, prepared_data: Dict[str, Any]) -> float:
        """Calculate silhouette score."""
        try:
            features = prepared_data['features']
            if len(features) == 0 or len(np.unique(solution)) < 2:
                return 0.0
            return silhouette_score(features, solution)
        except Exception:
            return 0.0

    def _calculate_calinski_harabasz_score(self, solution: np.ndarray, prepared_data: Dict[str, Any]) -> float:
        """Calculate Calinski-Harabasz score."""
        try:
            features = prepared_data['features']
            if len(features) == 0 or len(np.unique(solution)) < 2:
                return 0.0
            return calinski_harabasz_score(features, solution)
        except Exception:
            return 0.0

    def _calculate_davies_bouldin_score(self, solution: np.ndarray, prepared_data: Dict[str, Any]) -> float:
        """Calculate Davies-Bouldin score."""
        try:
            features = prepared_data['features']
            if len(features) == 0 or len(np.unique(solution)) < 2:
                return float('inf')
            return davies_bouldin_score(features, solution)
        except Exception:
            return float('inf')

    def _calculate_economic_significance(self, solution: np.ndarray, prepared_data: Dict[str, Any]) -> float:
        """Calculate economic significance score."""
        try:
            market_characteristics = prepared_data['market_characteristics']
            returns = market_characteristics.get('returns', np.zeros(len(solution)))

            unique_regimes = np.unique(solution)
            economic_scores = []

            for regime in unique_regimes:
                regime_mask = solution == regime
                regime_returns = returns[regime_mask]

                if len(regime_returns) > 0:
                    mean_return = np.mean(regime_returns)
                    volatility = np.std(regime_returns)
                    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                    economic_scores.append(abs(sharpe_ratio))
                else:
                    economic_scores.append(0.0)

            return np.mean(economic_scores) if economic_scores else 0.0

        except Exception:
            return 0.0

    def _calculate_trading_viability(self, solution: np.ndarray, prepared_data: Dict[str, Any]) -> float:
        """Calculate trading viability score."""
        try:
            # Calculate regime stability
            regime_changes = np.sum(solution[1:] != solution[:-1])
            stability = 1 - (regime_changes / len(solution))

            # Calculate volume liquidity
            market_characteristics = prepared_data['market_characteristics']
            volume = market_characteristics.get('volume', np.ones(len(solution)))
            liquidity = min(np.mean(volume) / 1000, 1.0)

            return (stability + liquidity) / 2

        except Exception:
            return 0.0

    def _calculate_cv_optimization(self, solution: np.ndarray, prepared_data: Dict[str, Any]) -> float:
        """Calculate CV optimization score."""
        try:
            market_characteristics = prepared_data['market_characteristics']
            returns = market_characteristics.get('returns', np.zeros(len(solution)))
            volume = market_characteristics.get('volume', np.ones(len(solution)))

            unique_regimes = np.unique(solution)
            cv_scores = []

            for regime in unique_regimes:
                regime_mask = solution == regime
                regime_returns = returns[regime_mask]
                regime_volume = volume[regime_mask]

                if len(regime_returns) > 0:
                    # Calculate CVs (removed momentum and entropy)
                    returns_cv = self._calculate_cv(regime_returns)
                    volume_cv = self._calculate_cv(regime_volume)
                    volatility_cv = self._calculate_cv(np.abs(regime_returns))

                    # Weighted CV score (lower is better)
                    weighted_cv = (
                        self.config.volatility_cv_weight * volatility_cv +
                        self.config.returns_cv_weight * returns_cv +
                        self.config.volume_cv_weight * volume_cv
                    )

                    cv_scores.append(1.0 / (1.0 + weighted_cv))  # Convert to maximization
                else:
                    cv_scores.append(0.0)

            return np.mean(cv_scores) if cv_scores else 0.0

        except Exception:
            return 0.0

    def _calculate_cv(self, data: np.ndarray) -> float:
        """Calculate coefficient of variation."""
        try:
            if len(data) == 0:
                return 0.0

            data = data[~np.isnan(data)]
            if len(data) == 0:
                return 0.0

            mean_val = np.mean(data)
            std_val = np.std(data)

            if mean_val == 0:
                return 0.0 if std_val == 0 else float('inf')

            return std_val / abs(mean_val)

        except Exception:
            return 0.0

    def _calculate_distribution_quality(self, solution: np.ndarray) -> float:
        """Calculate distribution quality score."""
        try:
            unique_regimes = np.unique(solution)
            n_regimes = len(unique_regimes)

            # Check cluster count target
            cluster_count_score = 0.0
            if self.config.target_cluster_count_min <= n_regimes <= self.config.target_cluster_count_max:
                cluster_count_score = 1.0
            else:
                if n_regimes < self.config.target_cluster_count_min:
                    penalty = (self.config.target_cluster_count_min - n_regimes) * 0.1
                else:
                    penalty = (n_regimes - self.config.target_cluster_count_max) * 0.1
                cluster_count_score = max(0.0, 1.0 - penalty)

            # Check distribution limits
            regime_counts = [np.sum(solution == regime) for regime in unique_regimes]
            total_count = len(solution)
            regime_percentages = [count / total_count for count in regime_counts]

            max_distribution_valid = all(p <= self.config.max_cluster_distribution for p in regime_percentages)
            min_distribution_valid = all(p >= self.config.min_cluster_distribution for p in regime_percentages)

            distribution_score = 0.0
            if max_distribution_valid:
                distribution_score += 0.5
            if min_distribution_valid:
                distribution_score += 0.5

            return (cluster_count_score + distribution_score) / 2

        except Exception:
            return 0.0

    def _calculate_temporal_consistency(self, solution: np.ndarray) -> float:
        """Calculate temporal consistency score."""
        try:
            # Calculate regime changes
            regime_changes = np.sum(solution[1:] != solution[:-1])
            stability = 1 - (regime_changes / len(solution))

            # Calculate regime durations
            durations = self._calculate_regime_durations(solution)
            avg_duration = np.mean(durations) if durations else 0
            duration_score = min(avg_duration / 50, 1.0)

            return (stability + duration_score) / 2

        except Exception:
            return 0.0

    def _calculate_regime_durations(self, solution: np.ndarray) -> List[int]:
        """Calculate regime durations."""
        try:
            durations = []
            current_regime = solution[0]
            current_duration = 1

            for i in range(1, len(solution)):
                if solution[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = solution[i]
                    current_duration = 1

            durations.append(current_duration)
            return durations

        except Exception:
            return []

    def _find_pareto_frontier(self, evaluated_solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto frontier from evaluated solutions."""
        try:
            if not evaluated_solutions:
                return []

            # Extract objectives
            objectives_list = []
            for solution in evaluated_solutions:
                objectives = solution['objectives']
                objectives_list.append([
                    objectives.get('silhouette_score', 0.0),
                    objectives.get('economic_significance', 0.0),
                    objectives.get('trading_viability', 0.0),
                    objectives.get('cv_optimization', 0.0),
                    objectives.get('distribution_quality', 0.0),
                    objectives.get('temporal_consistency', 0.0)
                ])

            # Find Pareto optimal solutions
            pareto_indices = self._find_pareto_optimal_indices(objectives_list)
            pareto_frontier = [evaluated_solutions[i] for i in pareto_indices]

            return pareto_frontier

        except Exception as e:
            self.logger.error(f"❌ Pareto frontier calculation failed: {e}")
            return []

    def _find_pareto_optimal_indices(self, objectives_list: List[List[float]]) -> List[int]:
        """Find indices of Pareto optimal solutions."""
        try:
            if not objectives_list:
                return []

            n_solutions = len(objectives_list)
            n_objectives = len(objectives_list[0])
            pareto_indices = []

            for i in range(n_solutions):
                is_pareto_optimal = True
                for j in range(n_solutions):
                    if i != j:
                        # Check if solution j dominates solution i
                        dominates = True
                        for k in range(n_objectives):
                            if objectives_list[j][k] < objectives_list[i][k]:
                                dominates = False
                                break

                        if dominates:
                            is_pareto_optimal = False
                            break

                if is_pareto_optimal:
                    pareto_indices.append(i)

            return pareto_indices

        except Exception:
            return []

    def _evolve_solutions(self, pareto_frontier: List[Dict[str, Any]],
                         prepared_data: Dict[str, Any]) -> List[np.ndarray]:
        """Evolve solutions through crossover and mutation."""
        try:
            new_solutions = []

            # Crossover between Pareto optimal solutions
            for i in range(len(pareto_frontier)):
                for j in range(i + 1, len(pareto_frontier)):
                    parent1 = pareto_frontier[i]['solution']
                    parent2 = pareto_frontier[j]['solution']

                    # Create offspring through crossover
                    offspring = self._crossover_solutions(parent1, parent2)
                    new_solutions.append(offspring)

            # Mutation of existing solutions
            for solution_data in pareto_frontier:
                solution = solution_data['solution']
                mutated = self._mutate_solution(solution)
                new_solutions.append(mutated)

            return new_solutions

        except Exception as e:
            self.logger.error(f"❌ Solution evolution failed: {e}")
            return []

    def _crossover_solutions(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two solutions."""
        try:
            offspring = np.zeros_like(parent1)
            crossover_point = len(parent1) // 2

            offspring[:crossover_point] = parent1[:crossover_point]
            offspring[crossover_point:] = parent2[crossover_point:]

            return offspring
        except Exception:
            return parent1

    def _mutate_solution(self, solution: np.ndarray) -> np.ndarray:
        """Mutate a solution."""
        try:
            mutated = solution.copy()
            mutation_rate = 0.1

            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    # Random mutation
                    mutated[i] = np.random.randint(0, len(np.unique(solution)))

            return mutated
        except Exception:
            return solution

    def _check_convergence(self, pareto_frontier: List[Dict[str, Any]], iteration: int) -> bool:
        """Check if optimization has converged."""
        try:
            if iteration < 10:  # Minimum iterations
                return False

            # Simple convergence check based on iteration count
            return iteration >= self.config.max_iterations

        except Exception:
            return False

    def _combine_objectives(self, objectives: Dict[str, float]) -> float:
        """Combine multiple objectives into a single score."""
        try:
            combined_score = (
                self.config.statistical_weight * objectives.get('silhouette_score', 0.0) +
                self.config.economic_weight * objectives.get('economic_significance', 0.0) +
                self.config.temporal_weight * objectives.get('temporal_consistency', 0.0) +
                self.config.cv_optimization_weight * objectives.get('cv_optimization', 0.0)
            )

            return combined_score

        except Exception:
            return 0.0

    def _evaluate_optimization_results(self, optimization_result: Dict[str, Any],
                                     prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate final optimization results."""
        try:
            pareto_frontier = optimization_result.get('pareto_frontier', [])

            if not pareto_frontier:
                return {'error': 'No Pareto frontier solutions found'}

            # Evaluate each solution in the Pareto frontier
            evaluated_solutions = []
            for solution_data in pareto_frontier:
                solution = solution_data['solution']
                objectives = self._evaluate_solution_objectives(solution, prepared_data)
                combined_score = self._combine_objectives(objectives)

                evaluated_solutions.append({
                    'solution': solution,
                    'objectives': objectives,
                    'combined_score': combined_score
                })

            # Sort by combined score
            evaluated_solutions.sort(key=lambda x: x['combined_score'], reverse=True)

            return {
                'evaluated_solutions': evaluated_solutions,
                'best_solution': evaluated_solutions[0] if evaluated_solutions else None,
                'optimization_quality': len(pareto_frontier)
            }

        except Exception as e:
            self.logger.error(f"❌ Optimization results evaluation failed: {e}")
            return {'error': str(e)}

    def _select_best_solution(self, optimization_result: Dict[str, Any],
                            final_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best solution from optimization results."""
        try:
            best_solution = final_evaluation.get('best_solution')

            if best_solution is None:
                return {'error': 'No best solution found'}

            return {
                'solution': best_solution['solution'],
                'objectives': best_solution['objectives'],
                'combined_score': best_solution['combined_score'],
                'selection_method': 'combined_score_optimization'
            }

        except Exception as e:
            self.logger.error(f"❌ Best solution selection failed: {e}")
            return {'error': str(e)}
