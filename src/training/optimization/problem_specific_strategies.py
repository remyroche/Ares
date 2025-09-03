from __future__ import annotations
'\nProblem-Specific Optimization Strategies\n\nThis module provides intelligent optimization strategies that automatically adapt\nto different problem characteristics:\n- Problem type detection\n- Adaptive strategy selection\n- Domain-specific optimizations\n- Constraint handling strategies\n- Multi-objective strategies\n'
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.logger import system_logger
import numpy as np

class ProblemType(Enum):
    """Enumeration of different problem types."""
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    MIXED = 'mixed'
    MULTI_OBJECTIVE = 'multi_objective'
    CONSTRAINED = 'constrained'
    NOISY = 'noisy'
    MULTI_MODAL = 'multi_modal'
    HIGH_DIMENSIONAL = 'high_dimensional'
    SPARSE = 'sparse'
    TIME_SERIES = 'time_series'

@dataclass
class ProblemCharacteristics:
    """Data class for problem characteristics."""
    problem_type: ProblemType
    dimensionality: int
    parameter_bounds: list[tuple[float, float]]
    is_noisy: bool
    is_multi_modal: bool
    has_constraints: bool
    is_multi_objective: bool
    sparsity_ratio: float
    correlation_structure: dict[str, float]
    complexity_score: float
    optimization_difficulty: str

class ProblemAnalyzer:
    """Analyzes optimization problems to determine their characteristics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('ProblemAnalyzer')

    def analyze_problem(self, objective_function: Callable, parameter_space: dict[str, Any], sample_points: np.ndarray | None=None, sample_values: np.ndarray | None=None) -> ProblemCharacteristics:
        """Analyze a problem to determine its characteristics."""
        dimensionality = len(parameter_space)
        parameter_bounds = self._extract_bounds(parameter_space)
        if sample_points is None or sample_values is None:
            sample_points, sample_values = self._generate_sample_data(objective_function, parameter_space)
        is_noisy = self._detect_noise(sample_values)
        is_multi_modal = self._detect_multi_modality(sample_points, sample_values)
        has_constraints = self._detect_constraints(parameter_space)
        is_multi_objective = self._detect_multi_objective(sample_values)
        sparsity_ratio = self._calculate_sparsity(sample_points)
        correlation_structure = self._analyze_correlations(sample_points, sample_values)
        complexity_score = self._calculate_complexity_score(dimensionality, is_noisy, is_multi_modal, sparsity_ratio)
        problem_type = self._determine_problem_type(parameter_space, is_multi_objective, has_constraints)
        optimization_difficulty = self._determine_difficulty(complexity_score)
        return ProblemCharacteristics(problem_type=problem_type, dimensionality=dimensionality, parameter_bounds=parameter_bounds, is_noisy=is_noisy, is_multi_modal=is_multi_modal, has_constraints=has_constraints, is_multi_objective=is_multi_objective, sparsity_ratio=sparsity_ratio, correlation_structure=correlation_structure, complexity_score=complexity_score, optimization_difficulty=optimization_difficulty)

    def _extract_bounds(self, parameter_space: dict[str, Any]) -> list[tuple[float, float]]:
        """Extract parameter bounds from parameter space."""
        bounds = []
        for param_config in parameter_space.values():
            if isinstance(param_config, dict):
                if 'min' in param_config and 'max' in param_config:
                    bounds.append((param_config['min'], param_config['max']))
                elif 'choices' in param_config:
                    choices = param_config['choices']
                    bounds.append((min(choices), max(choices)))
            elif isinstance(param_config, list | tuple) and len(param_config) == 2:
                bounds.append(tuple(param_config))
        return bounds

    def _generate_sample_data(self, objective_function: Callable, parameter_space: dict[str, Any], n_samples: int=100) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample data for problem analysis."""
        sample_points = []
        for _ in range(n_samples):
            point = {}
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    if 'min' in param_config and 'max' in param_config:
                        point[param_name] = np.random.uniform(param_config['min'], param_config['max'])
                    elif 'choices' in param_config:
                        point[param_name] = np.random.choice(param_config['choices'])
                elif isinstance(param_config, list | tuple) and len(param_config) == 2:
                    point[param_name] = np.random.uniform(param_config[0], param_config[1])
            sample_points.append(point)
        sample_values = []
        for point in sample_points:
            try:
                value = objective_function(point)
                sample_values.append(value)
            except Exception as e:
                self.logger.warning(f'Failed to evaluate point: {e}')
                sample_values.append(np.nan)
        sample_points_array = np.array([[point[param] for param in parameter_space] for point in sample_points])
        sample_values_array = np.array(sample_values)
        return (sample_points_array, sample_values_array)

    def _detect_noise(self, values: np.ndarray) -> bool:
        """Detect if the objective function is noisy."""
        if len(values) < 10:
            return False
        valid_values = values[~np.isnan(values)]
        if len(valid_values) < 5:
            return False
        sorted_indices = np.argsort(valid_values)
        sorted_values = valid_values[sorted_indices]
        differences = np.diff(sorted_values)
        noise_threshold = np.std(valid_values) * 0.1
        noisy_ratio = np.sum(np.abs(differences) < noise_threshold) / len(differences)
        return noisy_ratio > 0.3

    def _detect_multi_modality(self, points: np.ndarray, values: np.ndarray) -> bool:
        """Detect if the problem has multiple local optima."""
        if len(values) < 20:
            return False
        valid_mask = ~np.isnan(values)
        valid_points = points[valid_mask]
        valid_values = values[valid_mask]
        if len(valid_values) < 10:
            return False
        try:
            scaler = StandardScaler()
            normalized_points = scaler.fit_transform(valid_points)
            best_score = -np.inf
            best_n_clusters = 1
            for n_clusters in range(2, min(6, len(valid_values) // 5)):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(normalized_points)
                cluster_values = []
                for i in range(n_clusters):
                    cluster_mask = cluster_labels == i
                    if np.sum(cluster_mask) > 1:
                        cluster_values.append(np.mean(valid_values[cluster_mask]))
                if len(cluster_values) > 1:
                    separation = np.std(cluster_values)
                    score = separation / (n_clusters - 1)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            return best_n_clusters > 1 and best_score > np.std(valid_values) * 0.5
        except Exception as e:
            self.logger.warning(f'Error in multi-modality detection: {e}')
            return False

    def _detect_constraints(self, parameter_space: dict[str, Any]) -> bool:
        """Detect if the problem has constraints."""
        constraint_indicators = ['constraint', 'bound', 'limit', 'range']
        for param_name in parameter_space:
            if any((indicator in param_name.lower() for indicator in constraint_indicators)):
                return True
        for param_config in parameter_space.values():
            if isinstance(param_config, dict):
                if 'constraints' in param_config or 'dependencies' in param_config:
                    return True
        return False

    def _detect_multi_objective(self, values: np.ndarray) -> bool:
        """Detect if the problem is multi-objective."""
        if values.ndim > 1 and values.shape[1] > 1:
            return True
        if len(values) > 0:
            first_value = values[0]
            if isinstance(first_value, list | tuple) and len(first_value) > 1:
                return True
        return False

    def _calculate_sparsity(self, points: np.ndarray) -> float:
        """Calculate sparsity ratio of the parameter space."""
        if points.size == 0:
            return 0.0
        variances = np.var(points, axis=0)
        mean_variance = np.mean(variances)
        threshold = mean_variance * 0.1
        active_params = np.sum(variances > threshold)
        return active_params / points.shape[1]

    def _analyze_correlations(self, points: np.ndarray, values: np.ndarray) -> dict[str, float]:
        """Analyze correlations between parameters and objective values."""
        correlations = {}
        valid_mask = ~np.isnan(values)
        valid_points = points[valid_mask]
        valid_values = values[valid_mask]
        if len(valid_values) < 5:
            return correlations
        for i in range(valid_points.shape[1]):
            try:
                pearson_corr, _ = pearsonr(valid_points[:, i], valid_values)
                correlations[f'pearson_param_{i}'] = pearson_corr
                spearman_corr, _ = spearmanr(valid_points[:, i], valid_values)
                correlations[f'spearman_param_{i}'] = spearman_corr
            except Exception as e:
                self.logger.warning(f'Error calculating correlation for param {i}: {e}')
        return correlations

    def _calculate_complexity_score(self, dimensionality: int, is_noisy: bool, is_multi_modal: bool, sparsity_ratio: float) -> float:
        """Calculate overall problem complexity score."""
        score = 0.0
        score += min(dimensionality / 10.0, 1.0) * 0.3
        if is_noisy:
            score += 0.2
        if is_multi_modal:
            score += 0.3
        score += (1.0 - sparsity_ratio) * 0.2
        return min(score, 1.0)

    def _determine_problem_type(self, parameter_space: dict[str, Any], is_multi_objective: bool, has_constraints: bool) -> ProblemType:
        """Determine the specific problem type."""
        if is_multi_objective:
            return ProblemType.MULTI_OBJECTIVE
        if has_constraints:
            return ProblemType.CONSTRAINED
        has_discrete = False
        for param_config in parameter_space.values():
            if isinstance(param_config, dict) and 'choices' in param_config:
                has_discrete = True
                break
        if has_discrete:
            return ProblemType.DISCRETE
        return ProblemType.CONTINUOUS

    def _determine_difficulty(self, complexity_score: float) -> str:
        """Determine optimization difficulty based on complexity score."""
        if complexity_score < 0.3:
            return 'easy'
        if complexity_score < 0.7:
            return 'medium'
        return 'hard'

class BaseOptimizationStrategy(ABC):
    """Base class for optimization strategies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)

    @abstractmethod
    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt optimization strategy based on problem characteristics."""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""

class ContinuousOptimizationStrategy(BaseOptimizationStrategy):
    """Strategy for continuous optimization problems."""

    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt for continuous optimization."""
        adaptations = {'surrogate_model_type': 'gaussian_process', 'acquisition_function': 'expected_improvement', 'sampling_strategy': 'latin_hypercube', 'exploration_balance': 0.3, 'uncertainty_threshold': 0.1}
        if problem_characteristics.dimensionality > 20:
            adaptations['surrogate_model_type'] = 'random_forest'
            adaptations['sampling_strategy'] = 'random'
        if problem_characteristics.is_noisy:
            adaptations['surrogate_model_type'] = 'random_forest'
            adaptations['uncertainty_threshold'] = 0.2
        if problem_characteristics.is_multi_modal:
            adaptations['acquisition_function'] = 'upper_confidence_bound'
            adaptations['exploration_balance'] = 0.5
        return adaptations

    def get_strategy_name(self) -> str:
        return 'continuous_optimization'

class DiscreteOptimizationStrategy(BaseOptimizationStrategy):
    """Strategy for discrete optimization problems."""

    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt for discrete optimization."""
        adaptations = {'surrogate_model_type': 'random_forest', 'acquisition_function': 'probability_improvement', 'sampling_strategy': 'random', 'exploration_balance': 0.4, 'uncertainty_threshold': 0.15}
        adaptations['surrogate_model_type'] = 'random_forest'
        adaptations['exploration_balance'] = 0.5
        return adaptations

    def get_strategy_name(self) -> str:
        return 'discrete_optimization'

class MultiObjectiveOptimizationStrategy(BaseOptimizationStrategy):
    """Strategy for multi-objective optimization problems."""

    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt for multi-objective optimization."""
        adaptations = {'surrogate_model_type': 'ensemble', 'acquisition_function': 'multi_objective_ei', 'sampling_strategy': 'pareto_frontier', 'exploration_balance': 0.4, 'uncertainty_threshold': 0.2, 'multi_objective_weights': [0.5, 0.5]}
        adaptations['surrogate_model_type'] = 'ensemble'
        adaptations['sampling_strategy'] = 'pareto_frontier'
        return adaptations

    def get_strategy_name(self) -> str:
        return 'multi_objective_optimization'

class ConstrainedOptimizationStrategy(BaseOptimizationStrategy):
    """Strategy for constrained optimization problems."""

    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt for constrained optimization."""
        adaptations = {'surrogate_model_type': 'gaussian_process', 'acquisition_function': 'constrained_ei', 'sampling_strategy': 'feasible_latin_hypercube', 'exploration_balance': 0.3, 'uncertainty_threshold': 0.15, 'constraint_handling': 'penalty_method'}
        adaptations['surrogate_model_type'] = 'gaussian_process'
        adaptations['acquisition_function'] = 'constrained_ei'
        return adaptations

    def get_strategy_name(self) -> str:
        return 'constrained_optimization'

class NoisyOptimizationStrategy(BaseOptimizationStrategy):
    """Strategy for noisy optimization problems."""

    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt for noisy optimization."""
        adaptations = {'surrogate_model_type': 'random_forest', 'acquisition_function': 'robust_ei', 'sampling_strategy': 'noise_aware', 'exploration_balance': 0.4, 'uncertainty_threshold': 0.25, 'noise_handling': 'robust_estimation'}
        adaptations['surrogate_model_type'] = 'random_forest'
        adaptations['uncertainty_threshold'] = 0.25
        return adaptations

    def get_strategy_name(self) -> str:
        return 'noisy_optimization'

class HighDimensionalOptimizationStrategy(BaseOptimizationStrategy):
    """Strategy for high-dimensional optimization problems."""

    def adapt_optimization(self, problem_characteristics: ProblemCharacteristics, surrogate_optimizer: Any) -> dict[str, Any]:
        """Adapt for high-dimensional optimization."""
        adaptations = {'surrogate_model_type': 'random_forest', 'acquisition_function': 'sparse_ei', 'sampling_strategy': 'sparse_random', 'exploration_balance': 0.5, 'uncertainty_threshold': 0.2, 'dimensionality_reduction': True, 'feature_selection': True}
        adaptations['surrogate_model_type'] = 'random_forest'
        adaptations['dimensionality_reduction'] = True
        return adaptations

    def get_strategy_name(self) -> str:
        return 'high_dimensional_optimization'

class StrategySelector:
    """Selects and applies appropriate optimization strategies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('StrategySelector')
        self.strategies = {ProblemType.CONTINUOUS: ContinuousOptimizationStrategy(config), ProblemType.DISCRETE: DiscreteOptimizationStrategy(config), ProblemType.MULTI_OBJECTIVE: MultiObjectiveOptimizationStrategy(config), ProblemType.CONSTRAINED: ConstrainedOptimizationStrategy(config), ProblemType.NOISY: NoisyOptimizationStrategy(config), ProblemType.HIGH_DIMENSIONAL: HighDimensionalOptimizationStrategy(config)}
        self.problem_analyzer = ProblemAnalyzer(config)

    def select_and_apply_strategy(self, objective_function: Callable, parameter_space: dict[str, Any], surrogate_optimizer: Any) -> dict[str, Any]:
        """Select and apply the best optimization strategy."""
        problem_characteristics = self.problem_analyzer.analyze_problem(objective_function, parameter_space)
        self.logger.info(f'Problem characteristics: {problem_characteristics}')
        primary_strategy = self._select_primary_strategy(problem_characteristics)
        adaptations = primary_strategy.adapt_optimization(problem_characteristics, surrogate_optimizer)
        secondary_adaptations = self._apply_secondary_strategies(problem_characteristics)
        adaptations.update(secondary_adaptations)
        self.logger.info(f'Selected strategy: {primary_strategy.get_strategy_name()}')
        self.logger.info(f'Adaptations: {adaptations}')
        return adaptations

    def _select_primary_strategy(self, problem_characteristics: ProblemCharacteristics) -> BaseOptimizationStrategy:
        """Select the primary optimization strategy."""
        if problem_characteristics.is_multi_objective:
            return self.strategies[ProblemType.MULTI_OBJECTIVE]
        if problem_characteristics.has_constraints:
            return self.strategies[ProblemType.CONSTRAINED]
        if problem_characteristics.is_noisy:
            return self.strategies[ProblemType.NOISY]
        if problem_characteristics.dimensionality > 20:
            return self.strategies[ProblemType.HIGH_DIMENSIONAL]
        if problem_characteristics.problem_type == ProblemType.DISCRETE:
            return self.strategies[ProblemType.DISCRETE]
        return self.strategies[ProblemType.CONTINUOUS]

    def _apply_secondary_strategies(self, problem_characteristics: ProblemCharacteristics) -> dict[str, Any]:
        """Apply secondary strategies for additional adaptations."""
        adaptations = {}
        if problem_characteristics.is_noisy:
            adaptations.update({'noise_estimation': True, 'robust_kernel': True, 'multiple_evaluations': 3})
        if problem_characteristics.is_multi_modal:
            adaptations.update({'multi_start_optimization': True, 'restart_strategy': 'adaptive', 'local_search': True})
        if problem_characteristics.sparsity_ratio < 0.5:
            adaptations.update({'sparse_optimization': True, 'feature_importance': True, 'dimensionality_reduction': True})
        return adaptations