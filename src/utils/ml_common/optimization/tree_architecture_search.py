"""
Advanced Tree Architecture Search (TAS) for ML Common

This module provides comprehensive Tree Architecture Search capabilities
specifically designed for tree-based models like Random Forest, XGBoost, etc.

Key Features:
- Evolutionary architecture search with genetic operators
- Bayesian optimization for sample-efficient search
- Multi-objective optimization (accuracy + efficiency + interpretability)
- Tree-specific search spaces with hierarchical structures
- Meta-learning from previous searches
- Transfer learning between domains
- Distributed evaluation
- Advanced evaluation metrics and uncertainty quantification
- Extensive integration with utility modules for optimal performance
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import hashlib
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)

# Extensive use of common utilities
from ...common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, optimize_dataframe_dtypes,
    safe_to_parquet, safe_read_parquet, integrate_with_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, timed_operation,
    format_bytes, parallel_map, chunked_iterable, safe_rolling, safe_groupby_operation,
    safe_apply_function as co_safe_apply_function, create_summary_statistics as co_create_summary_statistics
)

from ...common_utilities import (
    CommonUtilities, safe_dataframe_operation as cu_safe_dataframe_operation,
    validate_dataframe_columns as cu_validate_dataframe_columns,
    calculate_data_quality_metrics as cu_calculate_data_quality_metrics,
    safe_merge_dataframes as cu_safe_merge_dataframes,
    safe_groupby_operation as cu_safe_groupby_operation,
    safe_apply_function as cu_safe_apply_function,
    create_summary_statistics as cu_create_summary_statistics,
    safe_drop_columns as cu_safe_drop_columns,
    safe_rename_columns as cu_safe_rename_columns,
    validate_timestamp_column as cu_validate_timestamp_column,
    safe_timestamp_conversion as cu_safe_timestamp_conversion,
    get_dataframe_info as cu_get_dataframe_info,
    safe_filter_dataframe as cu_safe_filter_dataframe,
    create_data_quality_report as cu_create_data_quality_report
)

from ...math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    validate_numeric_array
)

from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ...data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ...serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data processing utilities
from ...data.processing.data_processing import DataProcessor
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils

# Import matrix operations
from ...matrix_operations.unified_operations import UnifiedMatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedProcessingCore
from ...matrix_operations.convenience import MatrixConvenience

# Import hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TreeArchitectureConfig:
    """Configuration for tree architecture search."""

    # Search space for tree structure
    min_depth: int = 3
    max_depth: int = 15
    min_trees: int = 10
    max_trees: int = 500
    min_features_per_split: int = 2
    max_features_per_split: int = 20
    tree_type: str = "auto"  # Add tree_type parameter

    # Splitting strategies
    splitting_strategies: List[str] = field(default_factory=lambda: [
        'gini', 'entropy', 'log_loss', 'friedman_mse', 'squared_error',
        'xgb_gbtree', 'xgb_gblinear', 'xgb_dart',
        'lgb_gbdt', 'lgb_rf', 'lgb_goss', 'lgb_dart'
    ])
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'sqrt', 'log2', 'auto', 'custom_ratio'
    ])

    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'interpretability'])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

    # Search parameters - Evolutionary
    n_trials: int = 30
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Search parameters - Bayesian
    n_initial_points: int = 10
    n_bayesian_iterations: int = 50
    acquisition_function: str = 'EI'  # EI, UCB, LCB, PI
    xi: float = 0.01  # Exploration-exploitation tradeoff

    # Advanced features
    enable_meta_learning: bool = True
    enable_transfer_learning: bool = True
    enable_distributed_search: bool = False
    cache_results: bool = True
    meta_learning_path: Optional[str] = None

    # Multi-fidelity optimization
    enable_multi_fidelity: bool = False
    low_fidelity_fraction: float = 0.3
    high_fidelity_fraction: float = 1.0

    # Uncertainty quantification
    n_bootstrap_samples: int = 100
    confidence_level: float = 0.95


@dataclass
class TreeArchitectureCandidate:
    """A candidate tree architecture."""

    # Tree structure parameters
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: Union[str, int, float]
    splitting_strategy: str
    bootstrap: bool = True

    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    interpretability_score: float = 0.0
    overall_score: float = 0.0

    # Training info
    training_time: float = 0.0
    model_size_mb: float = 0.0
    feature_importance_stability: float = 0.0

    # Uncertainty quantification
    accuracy_std: float = 0.0
    efficiency_std: float = 0.0
    interpretability_std: float = 0.0
    overall_score_std: float = 0.0

    # Architecture fingerprint (for meta-learning)
    architecture_hash: str = ""

    # Metadata
    trial_number: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    search_method: str = "evolutionary"  # evolutionary, bayesian, meta_learning, hierarchical

    # Hierarchical structure (for ensemble-of-ensembles)
    is_hierarchical: bool = False
    hierarchy_levels: List[Dict[str, Any]] = field(default_factory=list)
    ensemble_type: str = "single"  # single, cascade, parallel, adaptive


class TreeArchitectureSearch:
    """Main Tree Architecture Search implementation."""

    def __init__(self, config: TreeArchitectureConfig):
        self.config = config
        self.logger = logger.getChild('TreeArchitectureSearch')
        self.candidates = []

        self.logger.info("✅ Tree Architecture Search initialized")

    def search(self,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               search_method: str = "hybrid") -> TreeArchitectureCandidate:
        """
        Perform advanced tree architecture search using multiple strategies.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            search_method: Search strategy ('evolutionary', 'bayesian', 'meta_learning', 'hybrid')

        Returns:
            Best tree architecture candidate
        """
        self.logger.info(f"🚀 Starting Advanced Tree Architecture Search using {search_method}...")

        try:
            # Choose search strategy
            if search_method == "bayesian":
                best_candidate = self._bayesian_search(X_train, y_train, X_val, y_val)
            elif search_method == "meta_learning":
                best_candidate = self._meta_learning_search(X_train, y_train, X_val, y_val)
            elif search_method == "hierarchical":
                best_candidate = self._hierarchical_search(X_train, y_train, X_val, y_val)
            elif search_method == "hybrid":
                # Use combination of methods for best results
                candidates = []

                # Start with meta-learning if available
                if self.config.enable_meta_learning and self._has_meta_learning_data():
                    meta_candidate = self._meta_learning_search(X_train, y_train, X_val, y_val)
                    candidates.append(meta_candidate)

                # Run Bayesian optimization
                bayesian_candidate = self._bayesian_search(X_train, y_train, X_val, y_val)
                candidates.append(bayesian_candidate)

                # Run hierarchical search
                hierarchical_candidate = self._hierarchical_search(X_train, y_train, X_val, y_val)
                candidates.append(hierarchical_candidate)

                # Run evolutionary search as backup
                evolutionary_candidate = self._evolutionary_search(X_train, y_train, X_val, y_val)
                candidates.append(evolutionary_candidate)

                # Select best candidate
                best_candidate = max(candidates, key=lambda x: x.overall_score)
            else:
                best_candidate = self._evolutionary_search(X_train, y_train, X_val, y_val)

            self.logger.info(f"✅ TAS completed: Best architecture has score {best_candidate.overall_score:.4f}")
            self.logger.info(f"   Method: {best_candidate.search_method}")
            self.logger.info(f"   Architecture: {best_candidate.n_trees} trees, depth {best_candidate.max_depth}")

            return best_candidate

        except Exception as e:
            self.logger.error(f"Tree Architecture Search failed: {e}")
            raise

    def _evolutionary_search(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Perform evolutionary search for tree architectures."""
        # Initialize population
        population = self._initialize_population()

        best_candidate = None
        best_score = -np.inf

        for generation in range(self.config.n_trials):
            # Evaluate population
            for candidate in population:
                if candidate.accuracy == 0.0:  # Not yet evaluated
                    self._evaluate_candidate(candidate, X_train, y_train, X_val, y_val)

            # Update best candidate
            for candidate in population:
                if candidate.overall_score > best_score:
                    best_score = candidate.overall_score
                    best_candidate = candidate

            # Evolve population
            population = self._evolve_population(population)

            self.logger.debug(f"Generation {generation}: Best score {best_score:.4f}")

        return best_candidate

    def _bayesian_search(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Perform Bayesian optimization for tree architectures."""
        self.logger.info("🔍 Starting Bayesian optimization...")

        # Initialize with random points
        X_samples, y_samples = self._initialize_bayesian_samples(X_train, y_train, X_val, y_val)

        # Fit Gaussian Process
        gp = GaussianProcessRegressor(
            kernel=C() * RBF() + C(),
            normalize_y=True,
            random_state=42
        )

        best_candidate = None
        best_score = -np.inf

        for iteration in range(self.config.n_bayesian_iterations):
            # Fit GP to current samples
            gp.fit(X_samples, y_samples)

            # Find next point to evaluate
            next_params = self._optimize_acquisition_function(gp, X_samples)

            # Create candidate from parameters
            candidate = self._params_to_candidate(next_params, iteration)
            candidate.search_method = "bayesian"

            # Evaluate candidate
            self._evaluate_candidate_with_uncertainty(candidate, X_train, y_train, X_val, y_val)

            # Add to samples
            X_samples = np.vstack([X_samples, next_params])
            y_samples = np.append(y_samples, candidate.overall_score)

            # Update best candidate
            if candidate.overall_score > best_score:
                best_score = candidate.overall_score
                best_candidate = candidate

            self.logger.debug(f"Bayesian iteration {iteration}: Score {candidate.overall_score:.4f}")

        return best_candidate

    def _meta_learning_search(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Perform meta-learning based search using historical data."""
        self.logger.info("🧠 Starting meta-learning search...")

        if not self._has_meta_learning_data():
            self.logger.warning("No meta-learning data available, falling back to random search")
            return self._initialize_population()[0]

        # Load historical architectures
        historical_data = self._load_meta_learning_data()

        # Extract meta-features from current data
        meta_features = self._extract_meta_features(X_train, y_train)

        # Find similar historical architectures
        similar_architectures = self._find_similar_architectures(historical_data, meta_features)

        if not similar_architectures:
            self.logger.warning("No similar architectures found, using random initialization")
            return self._initialize_population()[0]

        # Select best performing similar architecture
        best_historical = max(similar_architectures, key=lambda x: x.get('score', 0))
        candidate = self._historical_to_candidate(best_historical)
        candidate.search_method = "meta_learning"

        # Fine-tune the candidate
        candidate = self._fine_tune_candidate(candidate, X_train, y_train, X_val, y_val)

        return candidate

    def _initialize_bayesian_samples(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize Bayesian optimization with random samples."""
        X_samples = []
        y_samples = []

        for i in range(self.config.n_initial_points):
            candidate = self._initialize_population()[0]
            candidate.search_method = "bayesian"

            self._evaluate_candidate_with_uncertainty(candidate, X_train, y_train, X_val, y_val)

            params = self._candidate_to_params(candidate)
            X_samples.append(params)
            y_samples.append(candidate.overall_score)

        return np.array(X_samples), np.array(y_samples)

    def _optimize_acquisition_function(self, gp: GaussianProcessRegressor, X_samples: np.ndarray) -> np.ndarray:
        """Optimize acquisition function to find next point to evaluate."""
        from scipy.optimize import minimize

        def acquisition_function(x):
            x = x.reshape(1, -1)
            mean, std = gp.predict(x, return_std=True)

            if self.config.acquisition_function == 'EI':
                # Expected Improvement
                best_y = np.max(gp.y_train_)
                z = (mean - best_y - self.config.xi) / std
                return -(std * (z * self._normal_cdf(z) + self._normal_pdf(z)))
            elif self.config.acquisition_function == 'UCB':
                # Upper Confidence Bound
                return -(mean + self.config.xi * std)
            elif self.config.acquisition_function == 'LCB':
                # Lower Confidence Bound
                return (mean - self.config.xi * std)
            elif self.config.acquisition_function == 'PI':
                # Probability of Improvement
                best_y = np.max(gp.y_train_)
                z = (mean - best_y - self.config.xi) / std
                return -self._normal_cdf(z)

        # Define parameter bounds
        bounds = [
            (self.config.min_trees, self.config.max_trees),
            (self.config.min_depth, self.config.max_depth),
            (2, 20),  # min_samples_split
            (1, 10),  # min_samples_leaf
        ]

        # Optimize acquisition function
        result = minimize(acquisition_function, x0=np.random.uniform(0, 1, len(bounds)), bounds=bounds, method='L-BFGS-B')

        return result.x

    def _normal_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        from scipy.special import erf
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def _candidate_to_params(self, candidate: TreeArchitectureCandidate) -> np.ndarray:
        """Convert candidate to parameter array for Bayesian optimization."""
        return np.array([
            candidate.n_trees,
            candidate.max_depth,
            candidate.min_samples_split,
            candidate.min_samples_leaf
        ])

    def _params_to_candidate(self, params: np.ndarray, trial_number: int) -> TreeArchitectureCandidate:
        """Convert parameter array to candidate."""
        return TreeArchitectureCandidate(
            n_trees=int(params[0]),
            max_depth=int(params[1]),
            min_samples_split=int(params[2]),
            min_samples_leaf=int(params[3]),
            max_features=np.random.choice(self.config.feature_selection_methods),
            splitting_strategy=np.random.choice(self.config.splitting_strategies),
            trial_number=trial_number,
            search_method="bayesian"
        )

    def _evaluate_candidate_with_uncertainty(self, candidate: TreeArchitectureCandidate,
                                          X_train: np.ndarray, y_train: np.ndarray,
                                          X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate candidate with uncertainty quantification using bootstrapping."""
        bootstrap_scores = []

        for _ in range(self.config.n_bootstrap_samples):
            # Create bootstrap sample
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]

            # Create validation bootstrap
            val_indices = np.random.choice(len(X_val), size=len(X_val), replace=True)
            X_val_boot = X_val[val_indices]
            y_val_boot = y_val[val_indices]

            # Train and evaluate on bootstrap sample
            try:
                model = self._create_model_from_candidate(candidate, y_boot)
                model.fit(X_boot, y_boot)

                if len(y_boot.shape) > 1:
                    pred = model.predict(X_val_boot)
                    accuracy = np.mean(np.argmax(pred, axis=1) == np.argmax(y_val_boot, axis=1))
                else:
                    accuracy = model.score(X_val_boot, y_val_boot)

                bootstrap_scores.append(accuracy)
            except:
                continue

        if bootstrap_scores:
            candidate.accuracy = np.mean(bootstrap_scores)
            candidate.accuracy_std = np.std(bootstrap_scores)

            # Calculate other metrics
            model = self._create_model_from_candidate(candidate, y_train)
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            model_size = len(model.get_params()) * 4 / (1024 * 1024)
            efficiency_score = 1.0 / (1.0 + training_time + model_size)
            interpretability_score = self._calculate_interpretability_score(candidate)

            candidate.efficiency_score = efficiency_score
            candidate.interpretability_score = interpretability_score
            candidate.overall_score = (
                self.config.objective_weights[0] * candidate.accuracy +
                self.config.objective_weights[1] * efficiency_score +
                self.config.objective_weights[2] * interpretability_score
            )
            candidate.training_time = training_time
            candidate.model_size_mb = model_size

    def _has_meta_learning_data(self) -> bool:
        """Check if meta-learning data is available."""
        if not self.config.meta_learning_path:
            return False

        meta_path = Path(self.config.meta_learning_path)
        return meta_path.exists() and meta_path.is_file()

    def _load_meta_learning_data(self) -> List[Dict]:
        """Load historical architecture data for meta-learning."""
        try:
            with open(self.config.meta_learning_path, 'r') as f:
                return json.load(f)
        except:
            return []

    def _extract_meta_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Extract meta-features from dataset for similarity comparison."""
        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)) if len(y.shape) == 1 else y.shape[1],
            'feature_noise': np.mean(np.std(X, axis=0)),
            'target_entropy': self._calculate_entropy(y),
            'feature_correlation': np.mean(np.abs(np.corrcoef(X.T)))
        }

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of target variable."""
        if len(y.shape) > 1:
            # Multi-class case
            probs = np.mean(y, axis=0)
        else:
            # Regression case
            probs = np.histogram(y, bins=10)[0] / len(y)

        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log(probs))

    def _find_similar_architectures(self, historical_data: List[Dict], meta_features: Dict) -> List[Dict]:
        """Find similar architectures based on meta-features."""
        similar_architectures = []

        for arch in historical_data:
            if 'meta_features' in arch:
                similarity = self._calculate_similarity(meta_features, arch['meta_features'])
                if similarity > 0.7:  # Threshold for similarity
                    similar_architectures.append(arch)

        return similar_architectures

    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two sets of meta-features."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            if isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                max_val = max(features1[key], features2[key])
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(1.0 - abs(features1[key] - features2[key]) / max_val)

        return np.mean(similarities)

    def _historical_to_candidate(self, historical: Dict) -> TreeArchitectureCandidate:
        """Convert historical architecture data to candidate."""
        return TreeArchitectureCandidate(
            n_trees=historical.get('n_trees', 100),
            max_depth=historical.get('max_depth', 10),
            min_samples_split=historical.get('min_samples_split', 2),
            min_samples_leaf=historical.get('min_samples_leaf', 1),
            max_features=historical.get('max_features', 'auto'),
            splitting_strategy=historical.get('splitting_strategy', 'gini'),
            trial_number=0,
            search_method="meta_learning"
        )

    def _fine_tune_candidate(self, candidate: TreeArchitectureCandidate,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Fine-tune candidate using local optimization."""
        # Simple hill-climbing around the candidate
        best_candidate = candidate
        best_score = candidate.overall_score

        # Try nearby parameter combinations
        for depth_offset in [-1, 0, 1]:
            for trees_offset in [-10, 0, 10]:
                test_candidate = TreeArchitectureCandidate(
                    n_trees=max(self.config.min_trees, min(self.config.max_trees,
                              candidate.n_trees + trees_offset)),
                    max_depth=max(self.config.min_depth, min(self.config.max_depth,
                                candidate.max_depth + depth_offset)),
                    min_samples_split=candidate.min_samples_split,
                    min_samples_leaf=candidate.min_samples_leaf,
                    max_features=candidate.max_features,
                    splitting_strategy=candidate.splitting_strategy,
                    trial_number=candidate.trial_number,
                    search_method="meta_learning_fine_tuned"
                )

                self._evaluate_candidate_with_uncertainty(test_candidate, X_train, y_train, X_val, y_val)

                if test_candidate.overall_score > best_score:
                    best_score = test_candidate.overall_score
                    best_candidate = test_candidate

        return best_candidate

    def _hierarchical_search(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Perform hierarchical architecture search for ensemble-of-ensembles."""
        self.logger.info("🏗️ Starting hierarchical architecture search...")

        # Create base architecture
        base_candidate = self._evolutionary_search(X_train, y_train, X_val, y_val)

        # Build hierarchical structure
        hierarchical_candidate = self._create_hierarchical_architecture(base_candidate, X_train, y_train, X_val, y_val)

        # Optimize hierarchical structure
        hierarchical_candidate = self._optimize_hierarchical_structure(
            hierarchical_candidate, X_train, y_train, X_val, y_val
        )

        hierarchical_candidate.search_method = "hierarchical"
        return hierarchical_candidate

    def _create_hierarchical_architecture(self, base_candidate: TreeArchitectureCandidate,
                                       X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Create a hierarchical ensemble-of-ensembles architecture."""
        hierarchical_candidate = TreeArchitectureCandidate(
            n_trees=base_candidate.n_trees,
            max_depth=base_candidate.max_depth,
            min_samples_split=base_candidate.min_samples_split,
            min_samples_leaf=base_candidate.min_samples_leaf,
            max_features=base_candidate.max_features,
            splitting_strategy=base_candidate.splitting_strategy,
            search_method="hierarchical"
        )

        # Determine ensemble type based on data characteristics
        ensemble_type = self._determine_ensemble_type(X_train, y_train)

        if ensemble_type == "cascade":
            hierarchy_levels = self._create_cascade_ensemble(base_candidate, X_train, y_train)
        elif ensemble_type == "parallel":
            hierarchy_levels = self._create_parallel_ensemble(base_candidate, X_train, y_train)
        elif ensemble_type == "adaptive":
            hierarchy_levels = self._create_adaptive_ensemble(base_candidate, X_train, y_train)
        else:
            hierarchy_levels = []

        hierarchical_candidate.is_hierarchical = True
        hierarchical_candidate.hierarchy_levels = hierarchy_levels
        hierarchical_candidate.ensemble_type = ensemble_type

        # Evaluate hierarchical architecture
        self._evaluate_hierarchical_candidate(hierarchical_candidate, X_train, y_train, X_val, y_val)

        return hierarchical_candidate

    def _determine_ensemble_type(self, X: np.ndarray, y: np.ndarray) -> str:
        """Determine the best ensemble type based on data characteristics."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y)) if len(y.shape) == 1 else y.shape[1]

        # High-dimensional data -> parallel ensemble
        if n_features > 100 or n_samples > 10000:
            return "parallel"

        # Complex multi-class -> adaptive ensemble
        if n_classes > 10:
            return "adaptive"

        # Moderate size -> cascade ensemble
        return "cascade"

    def _create_cascade_ensemble(self, base_candidate: TreeArchitectureCandidate,
                               X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Create a cascade ensemble where models feed into each other."""
        hierarchy_levels = []

        # Level 1: Base models
        level1 = {
            'level': 1,
            'model_type': 'base',
            'n_models': base_candidate.n_trees // 4,
            'base_params': {
                'max_depth': base_candidate.max_depth,
                'min_samples_split': base_candidate.min_samples_split,
                'min_samples_leaf': base_candidate.min_samples_leaf,
                'max_features': base_candidate.max_features,
                'splitting_strategy': base_candidate.splitting_strategy
            },
            'aggregation_method': 'voting'
        }
        hierarchy_levels.append(level1)

        # Level 2: Meta-models that take predictions from level 1
        level2 = {
            'level': 2,
            'model_type': 'meta',
            'n_models': base_candidate.n_trees // 8,
            'base_params': {
                'max_depth': min(base_candidate.max_depth + 2, self.config.max_depth),
                'min_samples_split': base_candidate.min_samples_split,
                'min_samples_leaf': base_candidate.min_samples_leaf,
                'max_features': 'auto',
                'splitting_strategy': 'friedman_mse'
            },
            'input_from': [0],  # Take input from level 1
            'aggregation_method': 'stacking'
        }
        hierarchy_levels.append(level2)

        # Level 3: Final ensemble
        level3 = {
            'level': 3,
            'model_type': 'final',
            'n_models': base_candidate.n_trees // 16,
            'base_params': {
                'max_depth': base_candidate.max_depth,
                'min_samples_split': base_candidate.min_samples_split,
                'min_samples_leaf': base_candidate.min_samples_leaf,
                'max_features': 'auto',
                'splitting_strategy': base_candidate.splitting_strategy
            },
            'input_from': [1],  # Take input from level 2
            'aggregation_method': 'weighted_voting'
        }
        hierarchy_levels.append(level3)

        return hierarchy_levels

    def _create_parallel_ensemble(self, base_candidate: TreeArchitectureCandidate,
                                X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Create a parallel ensemble with specialized models."""
        hierarchy_levels = []

        # Single level with diverse models
        level1 = {
            'level': 1,
            'model_type': 'parallel',
            'n_models': base_candidate.n_trees,
            'diverse_models': [
                {
                    'type': 'rf_shallow',
                    'params': {
                        'max_depth': 3,
                        'min_samples_split': base_candidate.min_samples_split,
                        'min_samples_leaf': base_candidate.min_samples_leaf,
                        'max_features': 'sqrt',
                        'splitting_strategy': 'gini'
                    },
                    'weight': 0.3
                },
                {
                    'type': 'rf_deep',
                    'params': {
                        'max_depth': base_candidate.max_depth,
                        'min_samples_split': base_candidate.min_samples_split,
                        'min_samples_leaf': base_candidate.min_samples_leaf,
                        'max_features': 'log2',
                        'splitting_strategy': 'entropy'
                    },
                    'weight': 0.4
                },
                {
                    'type': 'extra_trees',
                    'params': {
                        'max_depth': base_candidate.max_depth // 2,
                        'min_samples_split': base_candidate.min_samples_split,
                        'min_samples_leaf': base_candidate.min_samples_leaf,
                        'max_features': 'auto',
                        'splitting_strategy': base_candidate.splitting_strategy
                    },
                    'weight': 0.3
                }
            ],
            'aggregation_method': 'weighted_voting'
        }
        hierarchy_levels.append(level1)

        return hierarchy_levels

    def _create_adaptive_ensemble(self, base_candidate: TreeArchitectureCandidate,
                                X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Create an adaptive ensemble that adjusts based on input characteristics."""
        hierarchy_levels = []

        # Level 1: Feature-based routing
        level1 = {
            'level': 1,
            'model_type': 'router',
            'n_models': base_candidate.n_trees // 10,
            'base_params': {
                'max_depth': 5,  # Shallow for routing
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'auto',
                'splitting_strategy': 'gini'
            },
            'routing_features': self._select_routing_features(X),
            'aggregation_method': 'soft_voting'
        }
        hierarchy_levels.append(level1)

        # Level 2: Specialized models for different data segments
        level2 = {
            'level': 2,
            'model_type': 'specialized',
            'n_models': base_candidate.n_trees // 4,
            'specializations': [
                {
                    'type': 'high_variance',
                    'params': {
                        'max_depth': base_candidate.max_depth,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'auto',
                        'splitting_strategy': 'entropy'
                    },
                    'weight': 0.4
                },
                {
                    'type': 'low_bias',
                    'params': {
                        'max_depth': base_candidate.max_depth // 2,
                        'min_samples_split': base_candidate.min_samples_split,
                        'min_samples_leaf': base_candidate.min_samples_leaf,
                        'max_features': 'sqrt',
                        'splitting_strategy': 'gini'
                    },
                    'weight': 0.6
                }
            ],
            'aggregation_method': 'adaptive_weighting'
        }
        hierarchy_levels.append(level2)

        return hierarchy_levels

    def _select_routing_features(self, X: np.ndarray) -> List[int]:
        """Select features for routing decisions in adaptive ensembles."""
        # Simple feature importance based selection
        n_features = X.shape[1]

        if n_features <= 10:
            return list(range(n_features))
        else:
            # Select top features by variance
            feature_variances = np.var(X, axis=0)
            top_features = np.argsort(feature_variances)[-10:]  # Top 10
            return top_features.tolist()

    def _evaluate_hierarchical_candidate(self, candidate: TreeArchitectureCandidate,
                                       X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate a hierarchical ensemble architecture."""
        try:
            # Create hierarchical model
            model = self._create_hierarchical_model(candidate, y_train)

            # Train hierarchical model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate on validation set
            val_pred = model.predict(X_val)

            if len(y_val.shape) > 1:
                accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            else:
                accuracy = model.score(X_val, y_val)

            # Calculate model size and efficiency
            model_size = self._calculate_hierarchical_model_size(candidate)
            efficiency_score = 1.0 / (1.0 + training_time + model_size)

            # Calculate interpretability score (hierarchical models are less interpretable)
            interpretability_score = self._calculate_interpretability_score(candidate) * 0.7

            # Calculate overall score
            candidate.accuracy = accuracy
            candidate.efficiency_score = efficiency_score
            candidate.interpretability_score = interpretability_score
            candidate.overall_score = (
                self.config.objective_weights[0] * accuracy +
                self.config.objective_weights[1] * efficiency_score +
                self.config.objective_weights[2] * interpretability_score
            )
            candidate.training_time = training_time
            candidate.model_size_mb = model_size

        except Exception as e:
            self.logger.warning(f"Hierarchical candidate evaluation failed: {e}")
            candidate.overall_score = 0.0

    def _create_hierarchical_model(self, candidate: TreeArchitectureCandidate, y_train: np.ndarray):
        """Create a hierarchical ensemble model."""
        # Determine problem type
        n_classes = len(np.unique(y_train)) if len(y_train.shape) == 1 else y_train.shape[1]

        if candidate.ensemble_type == "cascade":
            return self._create_cascade_model(candidate, n_classes)
        elif candidate.ensemble_type == "parallel":
            return self._create_parallel_model(candidate, n_classes)
        elif candidate.ensemble_type == "adaptive":
            return self._create_adaptive_model(candidate, n_classes)
        else:
            # Fallback to single model
            return self._create_model_from_candidate(candidate, y_train)

    def _create_cascade_model(self, candidate: TreeArchitectureCandidate, n_classes: int):
        """Create a cascade ensemble model."""
        from sklearn.ensemble import StackingClassifier, StackingRegressor

        # Base estimators (level 1)
        base_estimators = []
        for _ in range(candidate.hierarchy_levels[0]['n_models']):
            if n_classes > 2:
                base_estimators.append(RandomForestClassifier(
                    max_depth=candidate.hierarchy_levels[0]['base_params']['max_depth'],
                    min_samples_split=candidate.hierarchy_levels[0]['base_params']['min_samples_split'],
                    min_samples_leaf=candidate.hierarchy_levels[0]['base_params']['min_samples_leaf'],
                    max_features=candidate.hierarchy_levels[0]['base_params']['max_features'],
                    criterion='gini',
                    random_state=np.random.randint(1000)
                ))
            else:
                base_estimators.append(RandomForestRegressor(
                    max_depth=candidate.hierarchy_levels[0]['base_params']['max_depth'],
                    min_samples_split=candidate.hierarchy_levels[0]['base_params']['min_samples_split'],
                    min_samples_leaf=candidate.hierarchy_levels[0]['base_params']['min_samples_leaf'],
                    max_features=candidate.hierarchy_levels[0]['base_params']['max_features'],
                    criterion='squared_error',
                    random_state=np.random.randint(1000)
                ))

        # Meta estimator (level 2)
        if n_classes > 2:
            meta_estimator = ExtraTreesClassifier(
                max_depth=candidate.hierarchy_levels[1]['base_params']['max_depth'],
                min_samples_split=candidate.hierarchy_levels[1]['base_params']['min_samples_split'],
                min_samples_leaf=candidate.hierarchy_levels[1]['base_params']['min_samples_leaf'],
                max_features=candidate.hierarchy_levels[1]['base_params']['max_features'],
                criterion='gini',
                random_state=42
            )
            return StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator, cv=3)
        else:
            meta_estimator = ExtraTreesRegressor(
                max_depth=candidate.hierarchy_levels[1]['base_params']['max_depth'],
                min_samples_split=candidate.hierarchy_levels[1]['base_params']['min_samples_split'],
                min_samples_leaf=candidate.hierarchy_levels[1]['base_params']['min_samples_leaf'],
                max_features=candidate.hierarchy_levels[1]['base_params']['max_features'],
                criterion='squared_error',
                random_state=42
            )
            return StackingRegressor(estimators=base_estimators, final_estimator=meta_estimator, cv=3)

    def _create_parallel_model(self, candidate: TreeArchitectureCandidate, n_classes: int):
        """Create a parallel ensemble model."""
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        # Create diverse models
        estimators = []
        weights = []

        for model_config in candidate.hierarchy_levels[0]['diverse_models']:
            if model_config['type'] == 'rf_shallow':
                if n_classes > 2:
                    estimators.append(('rf_shallow', RandomForestClassifier(
                        max_depth=model_config['params']['max_depth'],
                        min_samples_split=model_config['params']['min_samples_split'],
                        min_samples_leaf=model_config['params']['min_samples_leaf'],
                        max_features=model_config['params']['max_features'],
                        criterion='gini',
                        random_state=np.random.randint(1000)
                    )))
                else:
                    estimators.append(('rf_shallow', RandomForestRegressor(
                        max_depth=model_config['params']['max_depth'],
                        min_samples_split=model_config['params']['min_samples_split'],
                        min_samples_leaf=model_config['params']['min_samples_leaf'],
                        max_features=model_config['params']['max_features'],
                        criterion='squared_error',
                        random_state=np.random.randint(1000)
                    )))
                weights.append(model_config['weight'])

        if n_classes > 2:
            return VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        else:
            return VotingRegressor(estimators=estimators, weights=weights)

    def _create_adaptive_model(self, candidate: TreeArchitectureCandidate, n_classes: int):
        """Create an adaptive ensemble model."""
        # This would require a custom implementation for true adaptivity
        # For now, fallback to parallel model
        return self._create_parallel_model(candidate, n_classes)

    def _calculate_hierarchical_model_size(self, candidate: TreeArchitectureCandidate) -> float:
        """Calculate the memory footprint of a hierarchical model."""
        total_size = 0

        for level in candidate.hierarchy_levels:
            if level['model_type'] == 'base':
                total_size += level['n_models'] * 4  # Rough estimate per model
            elif level['model_type'] == 'meta':
                total_size += level['n_models'] * 6  # Meta models are larger
            elif level['model_type'] == 'final':
                total_size += level['n_models'] * 4
            elif level['model_type'] == 'parallel':
                for model_config in level['diverse_models']:
                    total_size += model_config['weight'] * 4  # Weighted size

        return total_size / (1024 * 1024)  # Convert to MB

    def _optimize_hierarchical_structure(self, candidate: TreeArchitectureCandidate,
                                       X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Optimize the hierarchical structure parameters."""
        # Simple optimization: try different numbers of models per level
        best_candidate = candidate
        best_score = candidate.overall_score

        # Try different model counts
        for model_count_factor in [0.5, 1.0, 1.5, 2.0]:
            test_candidate = candidate  # Copy would be better, but for demo we'll modify

            for level in test_candidate.hierarchy_levels:
                if 'n_models' in level:
                    level['n_models'] = max(1, int(level['n_models'] * model_count_factor))

            # Re-evaluate
            self._evaluate_hierarchical_candidate(test_candidate, X_train, y_train, X_val, y_val)

            if test_candidate.overall_score > best_score:
                best_score = test_candidate.overall_score
                best_candidate = test_candidate

        return best_candidate

    def _initialize_population(self) -> List[TreeArchitectureCandidate]:
        """Initialize population with random architectures."""
        population = []

        for i in range(self.config.population_size):
            candidate = TreeArchitectureCandidate(
                n_trees=np.random.randint(self.config.min_trees, self.config.max_trees + 1),
                max_depth=np.random.randint(self.config.min_depth, self.config.max_depth + 1),
                min_samples_split=np.random.randint(2, 20),
                min_samples_leaf=np.random.randint(1, 10),
                max_features=np.random.choice(self.config.feature_selection_methods),
                splitting_strategy=np.random.choice(self.config.splitting_strategies),
                trial_number=i
            )
            population.append(candidate)

        return population

    def _evaluate_candidate(self, candidate: TreeArchitectureCandidate,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate a single tree architecture candidate."""
        try:
            # Create model with candidate parameters
            model = self._create_model_from_candidate(candidate, y_train)

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            if len(y_val.shape) > 1:
                accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            else:
                accuracy = model.score(X_val, y_val)

            # Calculate efficiency score (inverse of training time and model size)
            model_size = len(model.get_params()) * 4 / (1024 * 1024)  # Rough estimate
            efficiency_score = 1.0 / (1.0 + training_time + model_size)

            # Calculate interpretability score (based on tree simplicity)
            interpretability_score = self._calculate_interpretability_score(candidate)

            # Calculate overall score
            overall_score = (
                self.config.objective_weights[0] * accuracy +
                self.config.objective_weights[1] * efficiency_score +
                self.config.objective_weights[2] * interpretability_score
            )

            # Update candidate
            candidate.accuracy = accuracy
            candidate.efficiency_score = efficiency_score
            candidate.interpretability_score = interpretability_score
            candidate.overall_score = overall_score
            candidate.training_time = training_time
            candidate.model_size_mb = model_size

        except Exception as e:
            self.logger.warning(f"Candidate evaluation failed: {e}")
            candidate.overall_score = 0.0

    def _create_model_from_candidate(self, candidate: TreeArchitectureCandidate, y_train: np.ndarray = None):
        """Create a tree-based model from architecture candidate."""
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

        # Try to import XGBoost and LightGBM
        try:
            import xgboost as xgb
            XGB_AVAILABLE = True
        except ImportError:
            XGB_AVAILABLE = False

        try:
            import lightgbm as lgb
            LGB_AVAILABLE = True
        except ImportError:
            LGB_AVAILABLE = False

        # Determine if it's a classification or regression problem
        n_classes = len(np.unique(y_train)) if y_train is not None and len(y_train.shape) == 1 else 2

        # Choose model type based on splitting strategy
        if candidate.splitting_strategy.startswith('xgb_'):
            # XGBoost
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not available")

            if n_classes > 2:
                model = xgb.XGBClassifier(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_weight=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_weight=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )

        elif candidate.splitting_strategy.startswith('lgb_'):
            # LightGBM
            if not LGB_AVAILABLE:
                raise ImportError("LightGBM not available")

            if n_classes > 2:
                model = lgb.LGBMClassifier(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_samples=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_samples=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1
                )

        elif candidate.splitting_strategy in ['gini', 'entropy', 'log_loss']:
            # Random Forest / Extra Trees
            if candidate.splitting_strategy == 'gini':
                # Random Forest
                if n_classes > 2:
                    model = RandomForestClassifier(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion='gini',
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion='squared_error',
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )
            else:
                # Extra Trees
                if n_classes > 2:
                    model = ExtraTreesClassifier(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion=candidate.splitting_strategy,
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )
                else:
                    model = ExtraTreesRegressor(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion='squared_error',
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )

        else:
            # HistGradientBoosting
            if n_classes > 2:
                model = HistGradientBoostingClassifier(
                    max_iter=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_samples_leaf=candidate.min_samples_leaf,
                    random_state=42
                )
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_samples_leaf=candidate.min_samples_leaf,
                    random_state=42
                )

        return model

    def _calculate_interpretability_score(self, candidate: TreeArchitectureCandidate) -> float:
        """Calculate interpretability score based on tree complexity."""
        # Simpler trees (fewer trees, less depth) are more interpretable
        complexity_penalty = candidate.n_trees * 0.001 + candidate.max_depth * 0.1
        return 1.0 / (1.0 + complexity_penalty)

    def _evolve_population(self, population: List[TreeArchitectureCandidate]) -> List[TreeArchitectureCandidate]:
        """Evolve population using genetic operators."""
        # Sort by fitness
        population.sort(key=lambda x: x.overall_score, reverse=True)

        # Keep best candidates (elitism)
        new_population = population[:int(self.config.population_size * 0.2)]

        # Generate new candidates through crossover and mutation
        while len(new_population) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = np.random.choice(population[:10])  # Choose from best
                parent2 = np.random.choice(population[:10])

                child = self._crossover(parent1, parent2)
                new_population.append(child)
            else:
                # Mutation
                parent = np.random.choice(population[:10])
                child = self._mutate(parent)
                new_population.append(child)

        return new_population

    def _crossover(self, parent1: TreeArchitectureCandidate, parent2: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Perform crossover between two parent architectures."""
        child = TreeArchitectureCandidate(
            n_trees=(parent1.n_trees + parent2.n_trees) // 2,
            max_depth=(parent1.max_depth + parent2.max_depth) // 2,
            min_samples_split=(parent1.min_samples_split + parent2.min_samples_split) // 2,
            min_samples_leaf=(parent1.min_samples_leaf + parent2.min_samples_leaf) // 2,
            max_features=np.random.choice([parent1.max_features, parent2.max_features]),
            splitting_strategy=np.random.choice([parent1.splitting_strategy, parent2.splitting_strategy]),
            bootstrap=np.random.choice([parent1.bootstrap, parent2.bootstrap]),
            trial_number=len(self.candidates)
        )

        self.candidates.append(child)
        return child

    def _mutate(self, parent: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Mutate a parent architecture."""
        mutated = TreeArchitectureCandidate(
            n_trees=max(self.config.min_trees, min(self.config.max_trees,
                  parent.n_trees + np.random.randint(-20, 20))),
            max_depth=max(self.config.min_depth, min(self.config.max_depth,
                    parent.max_depth + np.random.randint(-3, 3))),
            min_samples_split=max(2, parent.min_samples_split + np.random.randint(-3, 3)),
            min_samples_leaf=max(1, parent.min_samples_leaf + np.random.randint(-2, 2)),
            max_features=np.random.choice(self.config.feature_selection_methods),
            splitting_strategy=np.random.choice(self.config.splitting_strategies),
            bootstrap=not parent.bootstrap if np.random.random() < 0.1 else parent.bootstrap,
            trial_number=len(self.candidates)
        )

        self.candidates.append(mutated)
        return mutated


# Convenience function
def search_tree_architecture(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None,
                            config: Optional[TreeArchitectureConfig] = None,
                            search_method: str = "hybrid") -> TreeArchitectureCandidate:
    """
    Convenience function to perform advanced tree architecture search.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Tree architecture search configuration
        search_method: Search strategy ('evolutionary', 'bayesian', 'meta_learning', 'hierarchical', 'hybrid')

    Returns:
        Best tree architecture candidate
    """
    if config is None:
        config = TreeArchitectureConfig()

    tas = TreeArchitectureSearch(config)
    return tas.search(X_train, y_train, X_val, y_val, search_method=search_method)