"""Tree architecture search utilities used by the TAS training pipeline."""

import hashlib
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, r2_score
except ImportError:  # pragma: no cover - fallback for minimal environments
    accuracy_score = None
    r2_score = None

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
    from src.utils.nas_tas.bayesian_tpe_optimizer import (
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
    optimization_strategy: str = "grid_tpe"  # grid, tpe, grid_tpe, bayesian, bayesian_forest, evolutionary
    early_stopping_patience: int = 10
    tpe_backend: str = "optuna"
    tpe_enable_grid_warmup: bool = True
    random_state: int = 42
    
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
    
    # Candidate caching
    enable_candidate_cache: bool = True
    cache_filename: str = "evaluation_cache.json"

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
    metric_name: str = "accuracy"
    
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
        self.metric_tracker: Dict[str, Dict[str, float]] = {}
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_file: Optional[Path] = None
        self.task_type: Optional[str] = None


        # Setup hardware optimization
        self._setup_hardware_optimization()

        # Setup optimization utilities
        self._setup_optimization_utilities()

        # Create results directory
        ensure_directory(self.config.results_dir)

        # Load cached evaluations if enabled
        if self.config.enable_candidate_cache:
            self.cache_file = Path(self.config.results_dir) / self.config.cache_filename
            self._load_evaluation_cache()

        tprint_info("✅ Tree Architecture Search initialized with shared utilities")
    
    def _setup_hardware_optimization(self):
        """Setup M1 hardware optimization if available."""
        if self.config.enable_m1_optimization and M1_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                memory_optimizer = get_m1_memory_optimizer()
                self.memory_optimizer = memory_optimizer if getattr(memory_optimizer, "optimize_array", None) else None
                cpu_optimizer = get_m1_cpu_optimizer()
                self.cpu_optimizer = cpu_optimizer if getattr(cpu_optimizer, "get_optimal_worker_count", None) else None
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
        self.task_type = self._infer_task_type(y_train)


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
        elif self.config.optimization_strategy == "bayesian_forest":
            best_candidate = self._run_bayesian_search(
                X_train, y_train, X_val, y_val, backend_override="skopt_forest"
            )
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
        if self.memory_optimizer and getattr(self.memory_optimizer, "optimize_array", None):
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
        grid_candidates = self._generate_grid_candidates(n_trials)
        best_candidate = None
        best_score = -np.inf

        for trial, params in enumerate(grid_candidates):
            candidate = TreeArchitectureCandidate(**params)
            candidate.trial_number = trial
            candidate.search_method = "grid"

            self._evaluate_candidate(candidate, X_train, y_train, X_val, y_val)
            self.candidates.append(candidate)

            if candidate.overall_score > best_score:
                best_score = candidate.overall_score
                best_candidate = candidate
                tprint_debug(f"Grid trial {trial}: New best {best_score:.4f}")

        # If requested trials exceed the unique grid combinations, fill the remainder via random sampling
        remaining_trials = n_trials - len(grid_candidates)
        for offset in range(remaining_trials):
            trial = len(grid_candidates) + offset
            candidate = self._sample_random_candidate()
            candidate.trial_number = trial
            candidate.search_method = "grid-random"

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

        optimizer = self._get_tpe_optimizer(
            n_trials=n_trials,
            enable_grid=self.config.tpe_enable_grid_warmup,
        )
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
    
    def _generate_grid_candidates(self, total_trials: int) -> List[Dict[str, Any]]:
        """Generate unique parameter combinations for grid sampling."""

        parameter_specs = [
            {
                "name": "n_trees",
                "type": "int",
                "min": self.config.min_trees,
                "max": self.config.max_trees,
                "max_points": min(total_trials, 50)
            },
            {
                "name": "max_depth",
                "type": "int",
                "min": self.config.min_depth,
                "max": self.config.max_depth,
                "max_points": min(total_trials, 30)
            },
            {
                "name": "min_samples_split",
                "type": "int",
                "min": 2,
                "max": 20,
                "max_points": min(total_trials, 10)
            },
            {
                "name": "min_samples_leaf",
                "type": "int",
                "min": 1,
                "max": 10,
                "max_points": min(total_trials, 10)
            },
            {
                "name": "learning_rate",
                "type": "float",
                "min": 0.01,
                "max": 0.3,
                "max_points": min(total_trials, 25)
            },
            {
                "name": "subsample",
                "type": "float",
                "min": 0.7,
                "max": 1.0,
                "max_points": min(total_trials, 25)
            },
            {
                "name": "max_features",
                "type": "categorical",
                "values": ["auto", "sqrt", "log2"]
            }
        ]

        num_params = len(parameter_specs)
        if num_params == 0:
            return []

        base_size = max(1, int(np.ceil(total_trials ** (1 / num_params))))

        counts: List[int] = []
        max_counts: List[int] = []
        for spec in parameter_specs:
            if spec["type"] == "categorical":
                values = spec["values"]
                counts.append(min(len(values), base_size))
                max_counts.append(len(values))
            else:
                max_points = spec.get("max_points", total_trials)
                max_possible = max(1, min(max_points, total_trials))
                counts.append(min(base_size, max_possible))
                max_counts.append(max_possible)

        def product(values: List[int]) -> int:
            result = 1
            for value in values:
                result *= max(1, value)
            return result

        combination_target = max(1, total_trials)
        while product(counts) < combination_target:
            # Increase counts cyclically while respecting the max per parameter
            for idx in range(len(counts)):
                if counts[idx] < max_counts[idx]:
                    counts[idx] += 1
                    if product(counts) >= combination_target:
                        break
            else:
                break  # Cannot increase further

        axes: List[List[Any]] = []
        for spec, count in zip(parameter_specs, counts):
            if spec["type"] == "categorical":
                axes.append(spec["values"][:count])
            elif spec["type"] == "int":
                values = np.linspace(spec["min"], spec["max"], count, dtype=int)
                unique_values = sorted(set(int(v) for v in values))
                axes.append(unique_values)
            else:
                values = np.linspace(spec["min"], spec["max"], count)
                axes.append([float(v) for v in values])

        combinations = list(itertools.product(*axes))
        unique_combinations = combinations[:min(len(combinations), total_trials)]

        candidates: List[Dict[str, Any]] = []
        for values in unique_combinations:
            params = {}
            for spec, value in zip(parameter_specs, values):
                params[spec["name"]] = value
            candidates.append(params)

        return candidates

    def _candidate_parameters(self, candidate: TreeArchitectureCandidate) -> Dict[str, Any]:
        """Extract the hyper-parameter portion of a candidate."""
        return {
            "model_type": candidate.model_type,
            "n_trees": int(candidate.n_trees),
            "max_depth": int(candidate.max_depth) if candidate.max_depth is not None else None,
            "min_samples_split": int(candidate.min_samples_split),
            "min_samples_leaf": int(candidate.min_samples_leaf),
            "max_features": candidate.max_features,
            "learning_rate": float(candidate.learning_rate),
            "subsample": float(candidate.subsample),
        }

    def _candidate_cache_key(self, candidate: TreeArchitectureCandidate) -> str:
        """Create a stable cache key for the candidate."""
        snapshot = self._candidate_parameters(candidate)
        serialized = json.dumps(snapshot, sort_keys=True, default=str)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _apply_cached_results(self, candidate: TreeArchitectureCandidate, cached: Dict[str, Any]) -> None:
        """Populate candidate metrics from cached evaluation results."""
        candidate.accuracy = cached.get("accuracy", 0.0)
        candidate.metric_name = cached.get("metric_name", candidate.metric_name)
        candidate.efficiency_score = cached.get("efficiency_score", 0.0)
        candidate.interpretability_score = cached.get("interpretability_score", 0.0)
        candidate.overall_score = cached.get("overall_score", 0.0)
        candidate.training_time = cached.get("training_time", 0.0)
        candidate.model_size_mb = cached.get("model_size_mb", 0.0)

        if cached.get("training_time") is not None:
            self._update_metric_tracker("training_time", cached["training_time"])
        if cached.get("complexity") is not None:
            self._update_metric_tracker("complexity", cached["complexity"])
        if cached.get("params"):
            params = cached["params"]
            if "n_trees" in params:
                self._update_metric_tracker("n_trees", params["n_trees"])
            if "max_depth" in params:
                depth_value = params["max_depth"] if params["max_depth"] is not None else 1
                self._update_metric_tracker("max_depth", depth_value)

    def _store_in_cache(self, cache_key: str, candidate: TreeArchitectureCandidate) -> None:
        """Persist evaluation results for reuse in future sessions."""
        depth_value = candidate.max_depth if candidate.max_depth is not None else 1
        entry = {
            "accuracy": candidate.accuracy,
            "metric_name": candidate.metric_name,
            "efficiency_score": candidate.efficiency_score,
            "interpretability_score": candidate.interpretability_score,
            "overall_score": candidate.overall_score,
            "training_time": candidate.training_time,
            "model_size_mb": candidate.model_size_mb,
            "complexity": candidate.n_trees * (2 ** max(depth_value, 1)),
            "params": self._candidate_parameters(candidate),
        }

        self.evaluation_cache[cache_key] = entry
        self._save_evaluation_cache()

    def _load_evaluation_cache(self) -> None:
        """Load cached candidate evaluations from disk."""
        if not self.cache_file or not self.cache_file.exists():
            return

        try:
            data = safe_json_load(self.cache_file)
            if isinstance(data, dict):
                entries = data.get("entries", {}) if "entries" in data else data
                if isinstance(entries, dict):
                    self.evaluation_cache = entries
        except Exception as exc:  # pragma: no cover - defensive
            tprint_warning(f"Failed to load evaluation cache: {exc}")
            self.evaluation_cache = {}

    def _save_evaluation_cache(self) -> None:
        """Persist the evaluation cache to disk."""
        if not self.config.enable_candidate_cache or not self.cache_file:
            return

        try:
            payload = {"version": 1, "entries": self.evaluation_cache}
            safe_json_dump(payload, self.cache_file)
        except Exception as exc:  # pragma: no cover - defensive
            tprint_warning(f"Failed to save evaluation cache: {exc}")

    def _update_metric_tracker(self, name: str, value: float) -> None:
        """Track the min and max values observed for a metric."""
        stats = self.metric_tracker.setdefault(name, {"min": None, "max": None})
        current_min = stats.get("min")
        current_max = stats.get("max")

        stats["min"] = value if current_min is None else min(current_min, value)
        stats["max"] = value if current_max is None else max(current_max, value)

    def _normalize_metric(self, name: str, value: float) -> float:
        """Normalize a metric to the [0, 1] range using tracked extrema."""
        stats = self.metric_tracker.get(name)
        if not stats:
            return 0.0

        min_val = stats.get("min")
        max_val = stats.get("max")

        if min_val is None or max_val is None or np.isclose(max_val, min_val):
            return 0.0

        return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))

    def _compute_efficiency_score(self, training_time: float, complexity: float) -> float:
        """Compute an efficiency score using normalized training time and complexity."""
        time_norm = self._normalize_metric("training_time", training_time)
        complexity_norm = self._normalize_metric("complexity", complexity)
        score = 1.0 - 0.5 * time_norm - 0.5 * complexity_norm
        return float(np.clip(score, 0.0, 1.0))

    def _compute_interpretability_score(self, n_trees: int, max_depth: Union[int, float]) -> float:
        """Compute interpretability favouring simpler models."""
        trees_norm = self._normalize_metric("n_trees", n_trees)
        depth_norm = self._normalize_metric("max_depth", max_depth)
        score = 1.0 - 0.6 * trees_norm - 0.4 * depth_norm
        return float(np.clip(score, 0.0, 1.0))

    def _compute_primary_metric(self, model, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Calculate the task-appropriate primary performance metric."""
        if self.task_type == "classification":
            predictions = model.predict(X_val)
            if accuracy_score is not None:
                return float(accuracy_score(y_val, predictions))
            return float(model.score(X_val, y_val))

        predictions = model.predict(X_val)
        if r2_score is not None:
            return float(r2_score(y_val, predictions))
        return float(model.score(X_val, y_val))

    def _infer_task_type(self, y: np.ndarray) -> str:
        """Infer whether the target is for classification or regression."""
        values = np.asarray(y).ravel()
        if values.size == 0:
            return "regression"

        series = pd.Series(values)

        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
            return "classification"

        unique_values = np.unique(values)
        if pd.api.types.is_integer_dtype(series) and unique_values.size <= max(20, values.size // 10):
            return "classification"

        if unique_values.size <= 10 and not np.issubdtype(values.dtype, np.floating):
            return "classification"

        return "regression"
    
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
            learning_rate=float(np.random.uniform(0.01, 0.3)),
            subsample=float(np.random.uniform(0.7, 1.0)),
            max_features=np.random.choice(["auto", "sqrt", "log2"])
        )
    
    def _evaluate_candidate(self, candidate: TreeArchitectureCandidate,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate a candidate architecture."""
        params_key = self._candidate_cache_key(candidate)

        if self.config.enable_candidate_cache and params_key in self.evaluation_cache:
            cached = self.evaluation_cache[params_key]
            self._apply_cached_results(candidate, cached)
            return

        try:
            model = self._create_model(candidate)

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            depth_value = candidate.max_depth if candidate.max_depth is not None else 1
            complexity = candidate.n_trees * (2 ** max(depth_value, 1))

            performance_score = self._compute_primary_metric(model, X_val, y_val)

            self._update_metric_tracker("training_time", training_time)
            self._update_metric_tracker("complexity", complexity)
            self._update_metric_tracker("n_trees", candidate.n_trees)
            self._update_metric_tracker("max_depth", depth_value)

            efficiency_score = self._compute_efficiency_score(training_time, complexity)
            interpretability_score = self._compute_interpretability_score(candidate.n_trees, depth_value)

            overall_score = (
                self.config.accuracy_weight * performance_score +
                self.config.efficiency_weight * efficiency_score +
                self.config.interpretability_weight * interpretability_score
            )

            candidate.accuracy = performance_score
            candidate.metric_name = "accuracy" if self.task_type == "classification" else "r2"
            candidate.efficiency_score = efficiency_score
            candidate.interpretability_score = interpretability_score
            candidate.overall_score = overall_score
            candidate.training_time = training_time
            candidate.model_size_mb = complexity / 1_000_000

            if self.config.enable_candidate_cache:
                self._store_in_cache(params_key, candidate)

        except Exception as e:
            tprint_warning(f"Evaluation failed: {e}")
            candidate.overall_score = 0.0

    def _create_model(self, candidate: TreeArchitectureCandidate):
        """Create a model instance for the candidate."""
        if candidate.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            forest_cls = RandomForestClassifier if self.task_type == "classification" else RandomForestRegressor

            model_kwargs = dict(

                n_estimators=candidate.n_trees,
                max_depth=candidate.max_depth,
                min_samples_split=candidate.min_samples_split,
                min_samples_leaf=candidate.min_samples_leaf,
                max_features=candidate.max_features,
                random_state=42,
            )

            if self.task_type == "classification":
                model_kwargs["n_jobs"] = -1

            return forest_cls(**model_kwargs)

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

    def _get_tpe_optimizer(
        self,
        n_trials: int,
        enable_grid: bool,
        backend: Optional[str] = None,
    ) -> BayesianTPEOptimizer:
        """Return configured Bayesian TPE optimizer."""
        backend_to_use = backend or self.config.tpe_backend
        config = BayesianTPEConfig(
            n_trials=max(1, n_trials),
            enable_grid_search=enable_grid and self.config.tpe_enable_grid_warmup,
            timeout_seconds=None,
            enable_early_stopping=True,
            early_stopping_patience=self.config.early_stopping_patience,
            backend=backend_to_use,
            enable_parallel=self.config.enable_parallel_processing,
            max_workers=self.config.max_parallel_jobs,
            random_state=self.config.random_state
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

    def _run_bayesian_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        backend_override: Optional[str] = None,
    ) -> TreeArchitectureCandidate:
        """Run Bayesian optimization leveraging the shared optimizer."""
        if not TPE_AVAILABLE:
            tprint_warning("Bayesian optimizer unavailable; falling back to random search")
            return self._run_random_search(X_train, y_train, X_val, y_val)

        optimizer = self._get_tpe_optimizer(
            n_trials=self.config.n_trials,
            enable_grid=True,
            backend=backend_override
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
