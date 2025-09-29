"""Shared base engine for NAS and TAS search components."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...common_operations import (
    cleanup_m1_optimizers,
    get_m1_cpu_optimizer,
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_memory_usage,
    gpu_context,
    integrate_with_m1_optimizers,
    memory_checkpoint,
    optimize_memory,
)
from ...common_utilities import CommonUtilities
from ...data.klines_parquet import get_klines_manager
from ...math_validation import (
    MathValidation,
    safe_mean,
    safe_percentile,
    safe_std,
    validate_correlation_matrix,
    validate_finite,
)
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.unified_operations import MatrixOperations
from ...matrix_operations.vectorized_core import VectorizedCore
from ...serialization_utils import UniversalSerializer
from ...tprint import (
    tprint_debug,
    tprint_error,
    tprint_info,
    tprint_progress,
    tprint_success,
    tprint_timer,
    tprint_warning,
)
from ..ml_common.optimization.bayesian_entry_timing_optimizer import (
    BayesianEntryTimingOptimizer,
)
from ..ml_common.optimization.grid_utils import GridSearchOptimizer
from ..ml_common.optimization.hierarchical_hpo import HierarchicalHPO

logger = logging.getLogger(__name__)


class BaseSearchEngine:
    """Shared search engine that encapsulates the common NAS/TAS behaviour."""

    #: Default number of trials used to trigger memory optimisations.
    _MEMORY_OPTIMIZATION_FREQUENCY = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, engine_name: str) -> None:
        self.config = config or {}
        self.engine_name = engine_name
        self.logger = logger.getChild(engine_name)

        tprint_debug(f"🔧 Initialising {self.engine_name} utilities")
        self.common_ops = CommonUtilities()
        self.math_validator = MathValidation()
        self.klines_manager = get_klines_manager()
        self.serializer = UniversalSerializer()

        # Matrix helpers used by both engines
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_core = VectorizedCore()

        # Hardware integration
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration.get("success"):
            tprint_success("✅ M1 integration successful")
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            tprint_warning("⚠️ M1 integration unavailable, continuing without accelerators")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

        # Optimisers shared by NAS/TAS engines
        self.bayesian_optimizer = BayesianEntryTimingOptimizer()
        self.grid_optimizer = GridSearchOptimizer()
        self.hierarchical_hpo = HierarchicalHPO()

        # Runtime bookkeeping
        self.performance_metrics: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []

        # Allow subclasses to customise context names
        self.search_context_name = f"{self.engine_name.lower()}_search"
        self.evaluation_context_name = f"{self.engine_name.lower()}_evaluation"

        tprint_success(f"✅ {self.engine_name} base initialised")

    # ------------------------------------------------------------------
    # Shared search execution
    # ------------------------------------------------------------------

    def _get_search_context(self) -> str:
        return getattr(self, "search_context_name", f"{self.engine_name.lower()}_search")

    def _get_evaluation_context(self) -> str:
        return getattr(
            self,
            "evaluation_context_name",
            f"{self.engine_name.lower()}_evaluation",
        )

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        validated: Dict[str, Any] = {}
        for name, value in params.items():
            try:
                if isinstance(value, (int, float)):
                    validated[name] = validate_finite(value, name)
                else:
                    validated[name] = value
            except ValueError as exc:  # pragma: no cover - validation guard
                tprint_warning(f"⚠️ Invalid parameter {name}: {exc}")
        return validated

    def _validate_feature_matrix(self, feature_matrix: np.ndarray) -> bool:
        if feature_matrix.size == 0:
            return False
        if not validate_correlation_matrix(feature_matrix):
            tprint_warning("⚠️ Invalid feature matrix correlation structure")
            return False
        return True

    def _evaluate_candidate(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        *,
        evaluation_context: Optional[str] = None,
        **extra: Any,
    ) -> float:
        validated_params = self._validate_params(params)

        with memory_checkpoint("candidate_data_preparation"):
            feature_matrix = self._create_feature_matrix(data, **extra)
            if not self._validate_feature_matrix(feature_matrix):
                return 0.0

        context_name = evaluation_context or self._get_evaluation_context()
        with (
            gpu_context(context_name)
            if self.gpu_manager is not None
            else memory_checkpoint(context_name)
        ):
            score = self._compute_score(feature_matrix, validated_params, **extra)

        try:
            return float(validate_finite(score, "candidate_score"))
        except ValueError:
            return 0.0

    def _run_bayesian_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        extra_context = extra_context or {}
        trials: List[Dict[str, Any]] = []
        best_score = -np.inf
        best_params: Dict[str, Any] = {}

        self.bayesian_optimizer.configure(
            search_space=search_space, n_trials=n_trials, random_state=42
        )

        for trial_idx in range(n_trials):
            tprint_progress(trial_idx, n_trials, f"Bayesian TPE trial {trial_idx}")
            trial_params = self.bayesian_optimizer.suggest()
            score = self._evaluate_candidate(data, trial_params, **extra_context)

            trial_record = {
                "trial_idx": trial_idx,
                "params": trial_params,
                "score": score,
                "timestamp": time.time(),
            }
            trials.append(trial_record)

            if score > best_score:
                best_score = score
                best_params = trial_params.copy()
                tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

            self.bayesian_optimizer.update(trial_params, score)

            if trial_idx % self._MEMORY_OPTIMIZATION_FREQUENCY == 0:
                optimize_memory()

        tprint_success(f"✅ Bayesian search completed: {len(trials)} trials")
        return best_params, best_score, trials

    def _run_grid_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        extra_context = extra_context or {}
        trials: List[Dict[str, Any]] = []
        best_score = -np.inf
        best_params: Dict[str, Any] = {}

        grid_params: Iterable[Dict[str, Any]] = self.grid_optimizer.generate_grid(
            search_space, max_trials=n_trials
        )
        grid_params = list(grid_params)
        total_trials = len(grid_params)
        tprint_info(f"🔧 Grid search: {total_trials} parameter combinations")

        for trial_idx, params in enumerate(grid_params):
            tprint_progress(trial_idx, total_trials, f"Grid search trial {trial_idx}")
            score = self._evaluate_candidate(data, params, **extra_context)

            trial_record = {
                "trial_idx": trial_idx,
                "params": params,
                "score": score,
                "timestamp": time.time(),
            }
            trials.append(trial_record)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

            if trial_idx % self._MEMORY_OPTIMIZATION_FREQUENCY == 0:
                optimize_memory()

        tprint_success(f"✅ Grid search completed: {len(trials)} trials")
        return best_params, best_score, trials

    def _run_hierarchical_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        extra_context = extra_context or {}
        trials: List[Dict[str, Any]] = []
        best_score = -np.inf
        best_params: Dict[str, Any] = {}

        self.hierarchical_hpo.configure(
            search_space=search_space, n_trials=n_trials, hierarchy_levels=3
        )

        for trial_idx in range(n_trials):
            tprint_progress(
                trial_idx, n_trials, f"Hierarchical HPO trial {trial_idx}"
            )
            trial_params = self.hierarchical_hpo.suggest()
            score = self._evaluate_candidate(data, trial_params, **extra_context)

            trial_record = {
                "trial_idx": trial_idx,
                "params": trial_params,
                "score": score,
                "timestamp": time.time(),
            }
            trials.append(trial_record)

            if score > best_score:
                best_score = score
                best_params = trial_params.copy()
                tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

            self.hierarchical_hpo.update(trial_params, score)

            if trial_idx % self._MEMORY_OPTIMIZATION_FREQUENCY == 0:
                optimize_memory()

        tprint_success(f"✅ Hierarchical search completed: {len(trials)} trials")
        return best_params, best_score, trials

    def _calculate_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trials:
            return {}

        scores = np.array([trial["score"] for trial in trials], dtype=float)
        metrics = {
            "mean_score": safe_mean(scores),
            "std_score": safe_std(scores),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "median_score": safe_percentile(scores, 50.0),
            "q25_score": safe_percentile(scores, 25.0),
            "q75_score": safe_percentile(scores, 75.0),
        }
        return metrics

    @tprint_timer("Search Execution")
    def run_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        *,
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tprint_info(
            f"🔍 Starting {self.engine_name} search with {optimization_method}"
        )

        search_results: Dict[str, Any] = {
            "method": optimization_method,
            "n_trials": n_trials,
            "trials": [],
            "best_params": {},
            "best_score": -np.inf,
            "search_time": 0.0,
            "performance_metrics": {},
        }

        start_time = time.time()
        context_name = self._get_search_context()
        context_manager = (
            gpu_context(context_name)
            if self.gpu_manager is not None
            else memory_checkpoint(context_name)
        )

        with context_manager:
            if optimization_method == "bayesian_tpe":
                best_params, best_score, trials = self._run_bayesian_search(
                    data, search_space, n_trials, extra_context=extra_context
                )
            elif optimization_method == "grid":
                best_params, best_score, trials = self._run_grid_search(
                    data, search_space, n_trials, extra_context=extra_context
                )
            elif optimization_method == "hierarchical":
                best_params, best_score, trials = self._run_hierarchical_search(
                    data, search_space, n_trials, extra_context=extra_context
                )
            else:
                tprint_error(f"❌ Unknown optimization method: {optimization_method}")
                return search_results

        search_time = time.time() - start_time
        search_results.update(
            {
                "best_params": best_params,
                "best_score": best_score,
                "trials": trials,
                "search_time": search_time,
                "performance_metrics": self._calculate_metrics(trials),
            }
        )

        tprint_success(f"✅ Search completed in {search_time:.2f}s")
        tprint_info(f"🏆 Best score: {best_score:.4f}")

        self.search_history.append(search_results)
        self.performance_metrics = search_results["performance_metrics"]

        return search_results

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @tprint_timer("Results Serialization")
    def save_results(self, results: Dict[str, Any], filepath: str) -> bool:
        try:
            tprint_info(f"💾 Saving results to {filepath}")
            results_with_metadata = {
                "results": results,
                "metadata": {
                    "timestamp": time.time(),
                    "engine_name": self.engine_name,
                    "m1_integration": self.m1_integration,
                    "memory_usage": get_memory_usage(),
                },
            }
            success = self.serializer.save(results_with_metadata, filepath)
            if success:
                tprint_success(f"✅ Results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save results to {filepath}")
            return success
        except Exception as exc:  # pragma: no cover - safety net
            tprint_error(f"❌ Error saving results: {exc}")
            return False

    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        try:
            tprint_info(f"📂 Loading results from {filepath}")
            results = self.serializer.load(filepath)
            if results:
                tprint_success(f"✅ Results loaded successfully from {filepath}")
            else:
                tprint_error(f"❌ Failed to load results from {filepath}")
            return results
        except Exception as exc:  # pragma: no cover - safety net
            tprint_error(f"❌ Error loading results: {exc}")
            return None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        try:
            tprint_info(f"🧹 Cleaning up {self.engine_name} resources")
            cleanup_m1_optimizers()
            self.search_history.clear()
            tprint_success(f"✅ {self.engine_name} cleanup completed")
        except Exception as exc:  # pragma: no cover - cleanup should never block
            tprint_error(f"❌ Error during cleanup: {exc}")

    def __enter__(self) -> "BaseSearchEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Hooks implemented by subclasses
    # ------------------------------------------------------------------

    def _create_feature_matrix(
        self, data: pd.DataFrame, **extra: Any
    ) -> np.ndarray:  # pragma: no cover - abstract hook
        raise NotImplementedError

    def _compute_score(
        self, feature_matrix: np.ndarray, params: Dict[str, Any], **extra: Any
    ) -> float:  # pragma: no cover - abstract hook
        raise NotImplementedError
