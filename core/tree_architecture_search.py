"""
Comprehensive Tree Architecture Search (TAS) for ML Common
Integrates with shared utilities from src/utils/
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from pathlib import Path
import json

# Import shared utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    safe_divide, safe_log, safe_sqrt, validate_finite, validate_positive,
    get_current_datetime, optimize_dataframe_dtypes
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
    tprint_success, tprint_performance
)
from src.utils.math_validation import (
    safe_correlation, safe_mean, safe_std, safe_percentile
)
from src.utils.serialization_utils import JSONSerializer

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer,
        BayesianTPEConfig,
    )
    TPE_AVAILABLE = True
except ImportError:
    TPE_AVAILABLE = False
    tprint_warning("TPE optimizer not available")

# Import hardware optimization
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    M1_AVAILABLE = True
except ImportError:
    M1_AVAILABLE = False
    tprint_warning("M1 hardware optimization not available")

logger = logging.getLogger(__name__)


@dataclass
class TreeArchitectureConfig:
    """Configuration for tree architecture search."""
    
    # Search parameters
    n_trials: int = 50
    optimization_strategy: str = "grid_tpe"  # grid, tpe, grid_tpe, evolutionary
    early_stopping_patience: int = 10
    
    # Tree parameter ranges
    min_trees: int = 10
    max_trees: int = 500
    min_depth: int = 3
    max_depth: int = 15
    
    # Optimization weights
    accuracy_weight: float = 0.4
    efficiency_weight: float = 0.3
    interpretability_weight: float = 0.3
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    max_parallel_jobs: int = 4
    
    # Results
    save_results: bool = True
    results_dir: str = "tree_search_results"


@dataclass
class TreeArchitectureCandidate:
    """A candidate tree architecture."""
    
    # Architecture parameters
    n_trees: int = 100
    max_depth: int = 6
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, float] = "auto"
    learning_rate: float = 0.1
    subsample: float = 1.0
    
    # Model type
    model_type: str = "random_forest"
    
    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    interpretability_score: float = 0.0
    overall_score: float = 0.0
    
    # Training info
    training_time: float = 0.0
    model_size_mb: float = 0.0
    
    # Metadata
    trial_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    search_method: str = "unknown"


class TreeArchitectureSearch:
    """Main Tree Architecture Search implementation with shared utilities integration."""
    
    def __init__(self, config: TreeArchitectureConfig):
        self.config = config
        self.logger = logger.getChild('TreeArchitectureSearch')
        self.candidates: List[TreeArchitectureCandidate] = []
        self.serializer = JSONSerializer()
        self.task_type: str = "regression"
        self._last_tpe_params: Optional[Dict[str, Any]] = None

        # Setup hardware optimization
        self._setup_hardware_optimization()

        # Setup optimization utilities
        self._setup_optimization_utilities()
        
        # Create results directory
        ensure_directory(self.config.results_dir)
        
        tprint_info("✅ Tree Architecture Search initialized with shared utilities")
    
    def _setup_hardware_optimization(self):
        """Setup M1 hardware optimization if available."""
        if self.config.enable_m1_optimization and M1_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_info("🚀 M1 hardware optimization enabled")
            except Exception as e:
                tprint_warning(f"M1 optimization setup failed: {e}")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _setup_optimization_utilities(self):
        """Setup optimization utilities."""
        if TPE_AVAILABLE:
            self.tpe_optimizer: Optional[BayesianTPEOptimizer] = None
            tprint_info("🔍 TPE optimizer available")
        else:
            self.tpe_optimizer = None
    
    def search(self, X_train: np.ndarray, y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
        """
        Perform tree architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best tree architecture candidate
        """
        tprint_info(f"🚀 Starting Tree Architecture Search ({self.config.optimization_strategy})")
        
        start_time = time.time()
        
        # Validate inputs
        if not validate_finite(X_train).all():
            raise ValueError("Training data contains non-finite values")
        
        # Split validation data if not provided
        if X_val is None or y_val is None:
            X_val, y_val = self._create_validation_split(X_train, y_train)
        
        # Determine task type (regression vs classification)
        self.task_type = self._determine_task_type(y_train)

        # Optimize data for M1 if available
        if self.memory_optimizer:
            X_train = self._optimize_data_for_m1(X_train)
            X_val = self._optimize_data_for_m1(X_val)

        # Run optimization based on strategy
        if self.config.optimization_strategy == "grid_tpe":
            best_candidate = self._run_grid_tpe_search(X_train, y_train, X_val, y_val)
        elif self.config.optimization_strategy == "tpe":
            best_candidate = self._run_tpe_search(X_train, y_train, X_val, y_val)
        elif self.config.optimization_strategy == "grid":
            best_candidate = self._run_grid_search(X_train, y_train, X_val, y_val)
        elif self.config.optimization_strategy == "bayesian":
            best_candidate = self._run_bayesian_search(X_train, y_train, X_val, y_val)
        elif self.config.optimization_strategy == "evolutionary":
            best_candidate = self._run_evolutionary_search(X_train, y_train, X_val, y_val)
        else:
            tprint_warning(
                f"Unknown optimization strategy '{self.config.optimization_strategy}', "
                "falling back to grid+TPE search."
            )
            best_candidate = self._run_grid_tpe_search(X_train, y_train, X_val, y_val)
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        search_time = time.time() - start_time
        tprint_performance("Tree Architecture Search", search_time)
        tprint_success(f"Best architecture: {best_candidate.overall_score:.4f} score")
        
        return best_candidate
    
    def _create_validation_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create validation split from training data."""
        n_val = int(0.2 * len(X))
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        return X[val_indices], y[val_indices]
    
    def _optimize_data_for_m1(self, data: np.ndarray) -> np.ndarray:
        """Optimize data for M1 processing."""
        if self.memory_optimizer:
            try:
                return self.memory_optimizer.optimize_array(data)
            except:
                return data
        return data
    
    def _run_grid_tpe_search(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Run combined grid + TPE search."""
        tprint_info("Phase 1: Grid search exploration")
        
        # Grid search phase (30% of trials)
        grid_trials = int(0.3 * self.config.n_trials)
        best_grid = self._run_grid_search_phase(X_train, y_train, X_val, y_val, grid_trials)
        
        tprint_info("Phase 2: TPE optimization")
        
        # TPE phase (70% of trials)
        tpe_trials = self.config.n_trials - grid_trials
        best_tpe = self._run_tpe_search_phase(X_train, y_train, X_val, y_val, tpe_trials, best_grid)
        
        # Return best overall
        return best_tpe if best_tpe.overall_score > best_grid.overall_score else best_grid
    
    def _run_grid_search_phase(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray, n_trials: int) -> TreeArchitectureCandidate:
        """Run grid search phase."""
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(n_trials):
            candidate = self._sample_grid_candidate(trial, n_trials)
            candidate.trial_number = trial
            candidate.search_method = "grid"
            
            self._evaluate_candidate(candidate, X_train, y_train, X_val, y_val)
            self.candidates.append(candidate)
            
            if candidate.overall_score > best_score:
                best_score = candidate.overall_score
                best_candidate = candidate
                tprint_debug(f"Grid trial {trial}: New best {best_score:.4f}")
        
        return best_candidate
    
    def _run_tpe_search_phase(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray, n_trials: int,
                             initial_best: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Run TPE search phase using the shared Bayesian optimizer."""
        if not TPE_AVAILABLE:
            tprint_warning("TPE optimizer unavailable, falling back to random search phase")
            return self._run_random_search(X_train, y_train, X_val, y_val)

        if n_trials <= 0:
            return initial_best

        tprint_info(f"Optimizing {n_trials} trials with Bayesian TPE")

        optimizer = self._get_tpe_optimizer(n_trials=n_trials, enable_grid=False)
        best_candidate = initial_best
        best_score = initial_best.overall_score
        def objective(params: Dict[str, Any],
                      X: np.ndarray,
                      y: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      **kwargs) -> float:
            nonlocal best_candidate, best_score

            candidate = self._params_to_candidate(params)
            candidate.trial_number = len(self.candidates)
            candidate.search_method = "tpe"

            self._evaluate_candidate(candidate, X, y, X_val, y_val)
            self.candidates.append(candidate)

            self._last_tpe_params = params.copy()

            if candidate.overall_score > best_score:
                best_score = candidate.overall_score
                best_candidate = candidate
                tprint_debug(f"TPE trial {candidate.trial_number}: New best {best_score:.4f}")

            return float(candidate.overall_score)

        optimizer.optimize(
            objective_function=objective,
            search_space=self._build_search_space(),
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val
        )

        return best_candidate
    
    def _sample_grid_candidate(self, trial: int, total_trials: int) -> TreeArchitectureCandidate:
        """Sample candidate using grid search strategy."""
        # Simple grid sampling
        grid_size = int(np.ceil(total_trials ** (1/3)))  # Cube root for 3D grid
        
        i = trial % grid_size
        j = (trial // grid_size) % grid_size
        k = (trial // (grid_size * grid_size)) % grid_size
        
        n_trees = int(self.config.min_trees + (self.config.max_trees - self.config.min_trees) * i / grid_size)
        max_depth = int(self.config.min_depth + (self.config.max_depth - self.config.min_depth) * j / grid_size)
        learning_rate = 0.01 + 0.29 * k / grid_size  # 0.01 to 0.3
        
        return TreeArchitectureCandidate(
            n_trees=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
    
    def _sample_tpe_candidate(self, trial: int) -> TreeArchitectureCandidate:
        """Sample candidate using the most recent TPE suggestion."""
        if self._last_tpe_params:
            return self._params_to_candidate(self._last_tpe_params)
        return self._sample_random_candidate()
    
    def _sample_random_candidate(self) -> TreeArchitectureCandidate:
        """Sample random candidate."""
        return TreeArchitectureCandidate(
            n_trees=np.random.randint(self.config.min_trees, self.config.max_trees + 1),
            max_depth=np.random.randint(self.config.min_depth, self.config.max_depth + 1),
            min_samples_split=np.random.randint(2, 21),
            min_samples_leaf=np.random.randint(1, 11),
            learning_rate=np.random.uniform(0.01, 0.3),
            subsample=np.random.uniform(0.7, 1.0),
            max_features=np.random.choice(["auto", "sqrt", "log2"])
        )
    
    def _evaluate_candidate(self, candidate: TreeArchitectureCandidate,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate a candidate architecture."""
        try:
            # Create and train model
            model = self._create_model(candidate)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate accuracy
            accuracy = model.score(X_val, y_val)
            
            # Calculate efficiency (inverse of time and complexity)
            complexity = candidate.n_trees * (2 ** candidate.max_depth)
            efficiency_score = 1.0 / (1.0 + training_time + complexity / 10000)
            
            # Calculate interpretability (simpler = more interpretable)
            interpretability_score = 1.0 / (1.0 + candidate.n_trees / 100 + candidate.max_depth / 10)
            
            # Calculate overall score
            overall_score = (
                self.config.accuracy_weight * accuracy +
                self.config.efficiency_weight * efficiency_score +
                self.config.interpretability_weight * interpretability_score
            )
            
            # Update candidate
            candidate.accuracy = accuracy
            candidate.efficiency_score = efficiency_score
            candidate.interpretability_score = interpretability_score
            candidate.overall_score = overall_score
            candidate.training_time = training_time
            candidate.model_size_mb = complexity / 1000000  # Rough estimate
            
        except Exception as e:
            tprint_warning(f"Evaluation failed: {e}")
            candidate.overall_score = 0.0

    def _create_model(self, candidate: TreeArchitectureCandidate):
        """Create model from candidate."""
        if candidate.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            model_cls = RandomForestRegressor
            if self.task_type == "classification":
                model_cls = RandomForestClassifier

            return model_cls(
                n_estimators=candidate.n_trees,
                max_depth=candidate.max_depth,
                min_samples_split=candidate.min_samples_split,
                min_samples_leaf=candidate.min_samples_leaf,
                max_features=candidate.max_features,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {candidate.model_type}")

    def _determine_task_type(self, y: np.ndarray) -> str:
        """Infer whether the problem is regression or classification."""
        try:
            series = pd.Series(y)
            if series.dtype.kind in {"O", "U", "S"}:
                return "classification"

            unique_values = series.dropna().unique()

            if series.dtype.kind in {"b"}:
                return "classification"

            if series.dtype.kind in {"i", "u"} and len(unique_values) <= min(20, max(2, len(series) // 10)):
                return "classification"

            if series.dtype.kind in {"f"}:
                if len(unique_values) <= min(10, max(2, len(series) // 15)):
                    # If float but effectively discrete (e.g., 0.0/1.0)
                    if np.allclose(unique_values, np.round(unique_values)):
                        return "classification"
        except Exception as e:
            tprint_warning(f"Could not determine task type, defaulting to regression. Error: {e}")

        return "regression"

    def _build_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Build search space definition for Bayesian/Evolutionary optimizers."""
        return {
            'n_trees': {
                'type': 'int',
                'low': self.config.min_trees,
                'high': self.config.max_trees
            },
            'max_depth': {
                'type': 'int',
                'low': self.config.min_depth,
                'high': self.config.max_depth
            },
            'min_samples_split': {
                'type': 'int',
                'low': 2,
                'high': 21
            },
            'min_samples_leaf': {
                'type': 'int',
                'low': 1,
                'high': 10
            },
            'learning_rate': {
                'type': 'float',
                'low': 0.01,
                'high': 0.3
            },
            'subsample': {
                'type': 'float',
                'low': 0.7,
                'high': 1.0
            },
            'max_features': {
                'type': 'categorical',
                'choices': ["auto", "sqrt", "log2"]
            }
        }

    def _params_to_candidate(self, params: Dict[str, Any]) -> TreeArchitectureCandidate:
        """Convert parameter dictionary to TreeArchitectureCandidate."""
        return TreeArchitectureCandidate(
            n_trees=int(params.get('n_trees', self.config.min_trees)),
            max_depth=int(params.get('max_depth', self.config.min_depth)),
            min_samples_split=int(params.get('min_samples_split', 2)),
            min_samples_leaf=int(params.get('min_samples_leaf', 1)),
            learning_rate=float(params.get('learning_rate', 0.1)),
            subsample=float(params.get('subsample', 1.0)),
            max_features=params.get('max_features', "auto")
        )

    def _candidate_to_params(self, candidate: TreeArchitectureCandidate) -> Dict[str, Any]:
        """Convert candidate to parameter dictionary."""
        return {
            'n_trees': candidate.n_trees,
            'max_depth': candidate.max_depth,
            'min_samples_split': candidate.min_samples_split,
            'min_samples_leaf': candidate.min_samples_leaf,
            'learning_rate': candidate.learning_rate,
            'subsample': candidate.subsample,
            'max_features': candidate.max_features
        }

    def _mutate_params(self, params: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """Mutate parameter dictionary."""
        mutated = params.copy()
        search_space = self._build_search_space()

        for key, config in search_space.items():
            if np.random.rand() > mutation_rate:
                continue

            if config['type'] == 'int':
                range_width = config['high'] - config['low']
                step = max(1, int(range_width * 0.05))  # Mutate by up to 5% of the range
                mutated[key] = int(np.clip(
                    mutated[key] + np.random.randint(-step, step + 1),
                    config['low'], config['high']))
            elif config['type'] == 'float':
                range_width = config['high'] - config['low']
                mutated[key] = float(np.clip(
                    mutated[key] + np.random.uniform(-0.1 * range_width, 0.1 * range_width),
                    config['low'],
                    config['high']
                ))
            elif config['type'] == 'categorical':
                choices = [choice for choice in config['choices'] if choice != mutated[key]]
                if choices:
                    mutated[key] = np.random.choice(choices)

        return mutated

    def _crossover_candidates(self, parent1: TreeArchitectureCandidate,
                               parent2: TreeArchitectureCandidate) -> Dict[str, Any]:
        """Perform uniform crossover between two parent candidates."""
        params1 = self._candidate_to_params(parent1)
        params2 = self._candidate_to_params(parent2)
        child_params = {}

        for key in params1:
            child_params[key] = params1[key] if np.random.rand() < 0.5 else params2[key]

        return child_params

    def _get_tpe_optimizer(self, n_trials: int, enable_grid: bool) -> BayesianTPEOptimizer:
        """Return configured Bayesian TPE optimizer."""
        config = BayesianTPEConfig(
            n_trials=max(1, n_trials),
            enable_grid_search=enable_grid,
            timeout_seconds=None,
            enable_early_stopping=True,
            early_stopping_patience=self.config.early_stopping_patience
        )
        self.tpe_optimizer = BayesianTPEOptimizer(config)
        return self.tpe_optimizer
    
    def _run_grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Run pure grid search."""
        return self._run_grid_search_phase(X_train, y_train, X_val, y_val, self.config.n_trials)
    
    def _run_tpe_search(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Run pure TPE search."""
        # Initialize with random candidate
        initial_candidate = self._sample_random_candidate()
        initial_candidate.trial_number = len(self.candidates)
        initial_candidate.search_method = "tpe"
        self._evaluate_candidate(initial_candidate, X_train, y_train, X_val, y_val)
        self.candidates.append(initial_candidate)

        remaining_trials = max(0, self.config.n_trials - 1)

        return self._run_tpe_search_phase(
            X_train, y_train, X_val, y_val, remaining_trials, initial_candidate
        )

    def _run_bayesian_search(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Run Bayesian optimization leveraging the shared optimizer."""
        if not TPE_AVAILABLE:
            tprint_warning("Bayesian optimizer unavailable; falling back to random search")
            return self._run_random_search(X_train, y_train, X_val, y_val)

        optimizer = self._get_tpe_optimizer(
            n_trials=self.config.n_trials,
            enable_grid=True
        )

        best_candidate: Optional[TreeArchitectureCandidate] = None
        best_score = -np.inf

        def objective(params: Dict[str, Any],
                      X: np.ndarray,
                      y: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      **kwargs) -> float:
            nonlocal best_candidate, best_score

            candidate = self._params_to_candidate(params)
            candidate.trial_number = len(self.candidates)
            candidate.search_method = "bayesian"

            self._evaluate_candidate(candidate, X, y, X_val, y_val)
            self.candidates.append(candidate)

            if candidate.overall_score > best_score:
                best_score = candidate.overall_score
                best_candidate = candidate

            return float(candidate.overall_score)

        optimizer.optimize(
            objective_function=objective,
            search_space=self._build_search_space(),
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val
        )

        if best_candidate is None:
            tprint_warning("Bayesian optimizer failed to improve; using random search result")
            return self._run_random_search(X_train, y_train, X_val, y_val)

        return best_candidate

    def _run_evolutionary_search(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Run a simple evolutionary strategy for tree search."""
        population_size = max(4, min(20, self.config.n_trials // 2 or 1))
        n_generations = max(1, self.config.n_trials // population_size)
        mutation_rate = 0.2
        elite_size = max(1, population_size // 5)

        population: List[TreeArchitectureCandidate] = [
            self._sample_random_candidate() for _ in range(population_size)
        ]

        best_candidate: Optional[TreeArchitectureCandidate] = None
        best_score = -np.inf

        total_evaluations = 0

        for generation in range(n_generations):
            tprint_info(f"Evolutionary generation {generation + 1}/{n_generations}")
            scored_population: List[Tuple[TreeArchitectureCandidate, float]] = []

            for candidate in population:
                candidate.trial_number = len(self.candidates)
                candidate.search_method = "evolutionary"
                self._evaluate_candidate(candidate, X_train, y_train, X_val, y_val)
                self.candidates.append(candidate)

                scored_population.append((candidate, candidate.overall_score))
                total_evaluations += 1

                if candidate.overall_score > best_score:
                    best_score = candidate.overall_score
                    best_candidate = candidate

                if total_evaluations >= self.config.n_trials:
                    break

            if total_evaluations >= self.config.n_trials:
                break

            scored_population.sort(key=lambda item: item[1], reverse=True)
            elites = [candidate for candidate, _ in scored_population[:elite_size]]

            weights = np.array([max(score, 1e-6) for _, score in scored_population])
            weights = weights / weights.sum() if weights.sum() > 0 else None

            next_population: List[TreeArchitectureCandidate] = elites.copy()

            while len(next_population) < population_size:
                parents = np.random.choice(
                    [candidate for candidate, _ in scored_population],
                    size=2,
                    replace=True,
                    p=weights
                ) if weights is not None else np.random.choice(
                    [candidate for candidate, _ in scored_population],
                    size=2,
                    replace=True
                )

                child_params = self._crossover_candidates(parents[0], parents[1])
                child_params = self._mutate_params(child_params, mutation_rate)
                next_population.append(self._params_to_candidate(child_params))

            population = next_population

        if best_candidate is None:
            return self._run_random_search(X_train, y_train, X_val, y_val)

        return best_candidate
    
    def _run_random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Run random search."""
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(self.config.n_trials):
            candidate = self._sample_random_candidate()
            candidate.trial_number = trial
            candidate.search_method = "random"
            
            self._evaluate_candidate(candidate, X_train, y_train, X_val, y_val)
            self.candidates.append(candidate)
            
            if candidate.overall_score > best_score:
                best_score = candidate.overall_score
                best_candidate = candidate
        
        return best_candidate
    
    def _save_results(self) -> None:
        """Save search results."""
        try:
            results_file = Path(self.config.results_dir) / f"tree_search_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.json"
            
            results_data = {
                'config': {
                    'n_trials': self.config.n_trials,
                    'optimization_strategy': self.config.optimization_strategy,
                    'min_trees': self.config.min_trees,
                    'max_trees': self.config.max_trees,
                    'min_depth': self.config.min_depth,
                    'max_depth': self.config.max_depth
                },
                'candidates': [
                    {
                        'trial_number': c.trial_number,
                        'n_trees': c.n_trees,
                        'max_depth': c.max_depth,
                        'learning_rate': c.learning_rate,
                        'accuracy': c.accuracy,
                        'efficiency_score': c.efficiency_score,
                        'interpretability_score': c.interpretability_score,
                        'overall_score': c.overall_score,
                        'training_time': c.training_time,
                        'search_method': c.search_method,
                        'timestamp': c.timestamp.isoformat()
                    }
                    for c in self.candidates
                ],
                'best_candidate': {
                    'overall_score': max(c.overall_score for c in self.candidates),
                    'candidate': next(c for c in self.candidates if c.overall_score == max(cc.overall_score for cc in self.candidates)).__dict__
                } if self.candidates else None
            }
            
            safe_json_dump(results_data, results_file)
            tprint_info(f"Results saved to {results_file}")
            
        except Exception as e:
            tprint_error(f"Failed to save results: {e}")


# Convenience function
def search_tree_architecture(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None,
                             y_val: Optional[np.ndarray] = None,
                             config: Optional[TreeArchitectureConfig] = None) -> TreeArchitectureCandidate:
    """
    Convenience function for tree architecture search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Search configuration
        
    Returns:
        Best tree architecture candidate
    """
    if config is None:
        config = TreeArchitectureConfig()
    
    search = TreeArchitectureSearch(config)
    return search.search(X_train, y_train, X_val, y_val)