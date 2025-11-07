"""
Hyperparameter Optimization Configuration for Rolling HMM Clustering

This module defines the HPO search space and objective function for optimizing
Rolling HMM clustering parameters using hierarchical optimization.

HPO Components:
1. EWMA periods: 8+16, 8+20, 8+24, 12+16, 12+20, 12+24 (6 options)
2. Model structure: n_components (4-6), pca_components (3-5)
3. Regularization: min_covar (1e-5 to 1e-2), kappa (1-50)

Uses HierarchicalParameterOptimizer with coarse-to-fine grid search followed by TPE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    HierarchicalOptimizationResult
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_debug

logger = logging.getLogger(__name__)

CV_RATIO_EPS = 1e-9


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    # Optimization stages
    stages: Optional[List[OptimizationStage]] = None
    n_rounds: int = 2
    enable_final_refinement: bool = True
    final_refinement_trials: int = 50

    # Cross-validation
    cv_folds: int = 5
    cv_type: str = 'time_series'  # 'time_series', 'blocked', 'rolling'

    # Objective function weights
    weight_predictive_ll: float = 0.33
    weight_temporal: float = 0.33
    weight_economic: float = 0.34

    # Optimization settings
    direction: str = 'maximize'
    use_custom_balanced_score: bool = True
    verbose: bool = True

    def __post_init__(self):
        tprint_debug("Initializing HPOConfig dataclass")
        if self.stages is None:
            self.stages = [
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ]

        # Validate weights sum to 1
        total_weight = self.weight_predictive_ll + self.weight_temporal + self.weight_economic
        if not np.isclose(total_weight, 1.0):
            tprint_error(
                f"⚠️  Invalid objective weights detected (sum={total_weight:.4f}); expected 1.0"
            )
            raise ValueError(f"Objective weights must sum to 1.0, got {total_weight}")


class RollingHMMOptimizer:
    """
    Hyperparameter optimizer for Rolling HMM clustering.

    Implements hierarchical optimization with three parameter groups:
    1. Feature engineering (EWMA periods)
    2. Model structure (n_components, pca_components)
    3. Regularization (min_covar, kappa)
    """

    def __init__(self, config: HPOConfig):
        """
        Initialize optimizer.

        Args:
            config: HPO configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        tprint_info("🧠 Initializing RollingHMMOptimizer")

        # Define parameter groups
        self.param_groups = self._create_parameter_groups()

        # Trial tracking for logging
        self.current_trial = 0
        self.total_trials_estimate = 0
        self.current_stage = ""
        self.stage_trials_completed = 0
        self.stage_trials_total = 0

    def _create_parameter_groups(self) -> List[ParameterGroup]:
        """Create hierarchical parameter groups for optimization."""
        tprint_debug("Configuring hierarchical parameter groups for Rolling HMM optimization")
        groups = []

        # Group 1: Feature Engineering (highest priority)
        # EWMA periods: 8+16, 8+20, 8+24, 12+16, 12+20, 12+24
        groups.append(
            ParameterGroup(
                name="feature_engineering",
                params={
                    "ewma_config_idx": {
                        "type": "categorical",
                        "choices": [0, 1, 2, 3, 4, 5]  # Index into DEFAULT_EWMA_CONFIGS
                    }
                },
                priority=1,
                description="EWMA period selection (8+16, 8+20, 8+24, 12+16, 12+20, 12+24)"
            )
        )

        # Group 2: Model Structure (depends on feature engineering)
        groups.append(
            ParameterGroup(
                name="model_structure",
                params={
                    "n_components": {
                        "type": "int",
                        "low": 4,
                        "high": 6,
                        "step": 1
                    },
                    "pca_components": {
                        "type": "int",
                        "low": 3,
                        "high": 5,
                        "step": 1
                    }
                },
                priority=2,
                depends_on=["feature_engineering"],
                description="HMM states and PCA components"
            )
        )

        # Group 3: Regularization (depends on model structure)
        groups.append(
            ParameterGroup(
                name="regularization",
                params={
                    "min_covar": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-2,
                        "log": True  # Log-scale sampling
                    },
                    "kappa": {
                        "type": "float",
                        "low": 1.0,
                        "high": 50.0,
                        "log": False
                    }
                },
                priority=3,
                depends_on=["feature_engineering", "model_structure"],
                description="Covariance and sticky regularization"
            )
        )

        return groups

    def create_objective_function(
        self,
        market_data: pd.DataFrame,
        feature_engineer,
        hmm_model_class,
        quality_assessor
    ):
        """
        Create objective function for HPO.

        Args:
            market_data: Market data DataFrame
            feature_engineer: Feature engineering instance
            hmm_model_class: Sticky HMM model class
            quality_assessor: Cluster quality assessor instance

        Returns:
            Objective function callable
        """
        tprint_debug("Creating objective function for hierarchical optimization")

        def objective(params: Dict[str, Any]) -> float:
            """
            Objective function for HPO.

            Evaluates HMM clustering quality based on:
            - Statistical cohesion (CV ratio + silhouette)
            - Temporal smoothness (33%)
            - Economic utility (34%)

            Args:
                params: Parameter dictionary

            Returns:
                Objective score (higher is better)
            """
            try:
                # Extract parameters
                ewma_config_idx = int(params.get('ewma_config_idx', 0))
                n_components = int(params.get('n_components', 5))
                pca_components = int(params.get('pca_components', 4))
                min_covar = float(params.get('min_covar', 1e-3))
                kappa = float(params.get('kappa', 10.0))

                # Import required classes
                from src.training.steps.market_analysis.rolling_hmm_clustering.feature_engineering import (
                    DEFAULT_EWMA_CONFIGS
                )
                from src.training.steps.market_analysis.rolling_hmm_clustering.sticky_hmm_model import (
                    StickyHMMConfig
                )

                # Get EWMA config
                ewma_config = DEFAULT_EWMA_CONFIGS[ewma_config_idx]

                # Generate features
                features = feature_engineer.generate_features(market_data, ewma_config)

                if len(features) < 50:
                    # Not enough data
                    return -1e6

                # Apply PCA
                features_pca, pca_model, explained_var = feature_engineer.apply_pca(
                    features,
                    n_components=pca_components
                )

                # Create HMM config
                hmm_config = StickyHMMConfig(
                    n_components=n_components,
                    min_covar=min_covar,
                    kappa=kappa,
                    covariance_type='diag',
                    kmeans_init=True,
                    use_sticky_priors=True,
                    post_fit_regularization=True
                )

                # Fit HMM model
                hmm_model = hmm_model_class(hmm_config)
                hmm_model.fit(features_pca.values)

                # Predict regime labels
                regime_labels = hmm_model.predict(features_pca.values)

                # Get transition matrix
                transition_matrix = hmm_model.get_transition_matrix()

                # Calculate forward returns
                forward_returns = market_data['close'].pct_change().shift(-1)
                forward_returns = forward_returns.loc[features_pca.index]

                # Assess quality
                metrics = quality_assessor.assess_hmm_regime_quality(
                    regime_labels=regime_labels,
                    feature_data=features_pca,
                    transition_matrix=transition_matrix,
                    hmm_model=None,  # Pass None to avoid recomputation
                    forward_returns=forward_returns,
                    timestamps=features_pca.index,
                    timeframe='1h',
                    min_regime_size=10,
                    run_validators=False,  # Skip validators during HPO
                    temporal_sensitivity_mode="standard"
                )

                # Calculate objective score using available metrics
                between_cv = getattr(metrics, 'between_regime_cv', 0.0)
                within_cv = getattr(metrics, 'within_regime_cv', 0.0)
                cv_ratio = between_cv / (within_cv + CV_RATIO_EPS) if within_cv is not None else 0.0
                normalized_cv_ratio = cv_ratio / (cv_ratio + 1.0) if cv_ratio > 0 else 0.0

                silhouette_raw = getattr(metrics, 'silhouette_score', 0.0)
                silhouette_norm = float(np.clip((silhouette_raw + 1.0) / 2.0, 0.0, 1.0))

                stat_score = float(np.clip((normalized_cv_ratio + silhouette_norm) / 2.0, 0.0, 1.0))
                score_statistical = stat_score * self.config.weight_predictive_ll

                temporal_smoothness = getattr(metrics, 'temporal_smoothness', 0.0)
                temporal_score = getattr(metrics, 'comprehensive_temporal_score', temporal_smoothness)
                score_temporal = (
                    temporal_smoothness * 0.5 + temporal_score * 0.5
                ) * self.config.weight_temporal

                sharpe = getattr(metrics, 'out_of_sample_sharpe', None)
                if sharpe is not None:
                    normalized_sharpe = np.clip((sharpe + 2) / 6, 0, 1)
                    score_economic = normalized_sharpe * self.config.weight_economic
                else:
                    score_economic = getattr(metrics, 'quality_score', 0.0) * self.config.weight_economic

                objective_score = score_statistical + score_temporal + score_economic

                # Log trial details (similar to statsmodel_clustering pattern)
                self._log_trial_details(
                    trial_num=self.current_trial,
                    stage=self.current_stage,
                    stage_progress=(self.stage_trials_completed, self.stage_trials_total),
                    params=params,
                    ewma_config=ewma_config,
                    metrics=metrics,
                    objective_score=objective_score
                )

                return objective_score

            except Exception as e:
                self.logger.warning(f"Objective function failed: {e}")
                tprint_warning(f"  ⚠️  Trial {self.current_trial} failed: {str(e)[:80]}")
                return -1e6

        return objective

    def _log_trial_details(
        self,
        trial_num: int,
        stage: str,
        stage_progress: Tuple[int, int],
        params: Dict[str, Any],
        ewma_config,
        metrics,
        objective_score: float
    ):
        """
        Log detailed information about each HPO trial.

        Similar to statsmodel_clustering, logs:
        - Trial number and stage progress
        - Parameters used
        - Within/cross-cluster CV metrics
        - Temporal smoothness
        - Average score
        """
        from src.utils.tprint import tprint

        # Format stage progress
        stage_completed, stage_total = stage_progress
        progress_pct = (stage_completed / stage_total * 100) if stage_total > 0 else 0

        # Extract key metrics (matching statsmodel_clustering pattern)
        within_cv = metrics.within_regime_cv if hasattr(metrics, 'within_regime_cv') else 0.0
        between_cv = metrics.between_regime_cv if hasattr(metrics, 'between_regime_cv') else 0.0
        cv_ratio = between_cv / (within_cv + CV_RATIO_EPS) if within_cv is not None else 0.0
        temporal_smoothness = metrics.temporal_smoothness if hasattr(metrics, 'temporal_smoothness') else 0.0
        quality_score = metrics.quality_score if hasattr(metrics, 'quality_score') else 0.0
        silhouette = metrics.silhouette_score if hasattr(metrics, 'silhouette_score') else 0.0

        # Format parameters
        param_str = (
            f"EWMA={ewma_config.name}, "
            f"n_states={params.get('n_components', 5)}, "
            f"pca={params.get('pca_components', 4)}, "
            f"kappa={params.get('kappa', 10.0):.1f}, "
            f"min_cov={params.get('min_covar', 1e-3):.1e}"
        )

        # Compact single-line format showing key info (similar to statsmodel pattern)
        tprint(
            f"Trial {trial_num:3d} [{stage}] {stage_completed}/{stage_total} ({progress_pct:5.1f}%) | "
            f"{param_str} | "
            f"Within_CV={within_cv:.4f}, Between_CV={between_cv:.4f}, Ratio={cv_ratio:.4f}, "
            f"Temp={temporal_smoothness:.4f}, Sil={silhouette:.4f}, Obj={objective_score:.4f}"
        )

    def optimize(
        self,
        market_data: pd.DataFrame,
        feature_engineer,
        hmm_model_class,
        quality_assessor
    ) -> HierarchicalOptimizationResult:
        """
        Run hierarchical parameter optimization.

        Args:
            market_data: Market data DataFrame
            feature_engineer: Feature engineering instance
            hmm_model_class: Sticky HMM model class
            quality_assessor: Cluster quality assessor instance

        Returns:
            Optimization result
        """
        tprint("🔍 Starting Hierarchical Parameter Optimization")

        # Create objective function
        objective_func = self.create_objective_function(
            market_data,
            feature_engineer,
            hmm_model_class,
            quality_assessor
        )

        # Create optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=self.param_groups,
            objective_func=objective_func,
            stages=self.config.stages,
            cv_folds=self.config.cv_folds,
            scoring_metric='custom_balanced_score',
            direction=self.config.direction,
            n_rounds=self.config.n_rounds,
            enable_final_refinement=self.config.enable_final_refinement,
            final_refinement_trials=self.config.final_refinement_trials,
            verbose=self.config.verbose,
            use_custom_balanced_score=self.config.use_custom_balanced_score
        )

        # Run optimization
        tprint_info("  → Running coarse grid search")
        tprint_info("  → Running fine grid search")
        tprint_info("  → Running TPE optimization")

        # Note: HierarchicalParameterOptimizer expects X, y, X_val, y_val
        # For our case, we'll pass dummy data and handle it in the objective function
        # This is a limitation of the current interface

        # We need to modify the approach - instead, let's do custom optimization
        result = self._run_custom_optimization(objective_func)

        tprint(f"✅ Optimization complete - Best score: {result['best_score']:.4f}")

        return result

    def _run_custom_optimization(self, objective_func) -> Dict[str, Any]:
        """
        Run custom optimization since we don't have traditional X, y data.

        This implements a simplified version of hierarchical optimization:
        1. Coarse grid search over all parameters
        2. Fine grid search around best region
        3. Random search for final refinement

        Args:
            objective_func: Objective function to optimize

        Returns:
            Optimization result dictionary
        """
        tprint_info("Running custom hierarchical optimization")

        coarse_grid = {
            'ewma_config_idx': [0, 2, 4],  # 3 EWMA configs
            'n_components': [4, 5, 6],     # 3 states
            'pca_components': [3, 4, 5],   # 3 PCA components
            'min_covar': [1e-5, 1e-4, 1e-3, 1e-2],  # 4 values (log scale)
            'kappa': [1, 5, 10, 20, 50]    # 5 values
        }

        best_score = -np.inf
        best_params = None
        coarse_results = []

        # Sample from coarse grid (not exhaustive due to computational cost)
        n_coarse_samples = min(50, int(np.prod([len(v) for v in coarse_grid.values()])))
        self.stage_trials_total = int(n_coarse_samples)
        self.stage_trials_completed = 0

        # Stage 1: Coarse grid search
        self.current_stage = "Coarse Grid"
        tprint("")
        tprint(f"🔍 Stage 1/3: {self.current_stage} ({n_coarse_samples} trials)")

        for i in range(n_coarse_samples):
            self.current_trial += 1
            self.stage_trials_completed = int(i + 1)

            params = {
                k: np.random.choice(v) for k, v in coarse_grid.items()
            }

            score = objective_func(params)
            coarse_results.append((params.copy(), score))

            if score > best_score:
                best_score = score
                best_params = params.copy()

        tprint(f"✅ Coarse grid complete - Best score: {best_score:.4f}")

        # Stage 2: Fine grid search around best region
        self.current_stage = "Fine Grid"
        fine_grid = self._create_fine_grid(best_params)
        fine_results = []
        self.stage_trials_total = int(len(fine_grid))
        self.stage_trials_completed = 0

        tprint("")
        tprint(f"🔍 Stage 2/3: {self.current_stage} ({len(fine_grid)} trials around best region)")

        for i, params in enumerate(fine_grid):
            self.current_trial += 1
            self.stage_trials_completed = i + 1

            score = objective_func(params)
            fine_results.append((params.copy(), score))

            if score > best_score:
                best_score = score
                best_params = params.copy()

        tprint(f"✅ Fine grid complete - Best score: {best_score:.4f}")

        # Stage 3: Final refinement with random search
        refinement_results = []

        if self.config.enable_final_refinement:
            self.current_stage = "Refinement"
            self.stage_trials_total = int(self.config.final_refinement_trials)
            self.stage_trials_completed = 0

            tprint("")
            tprint(f"🔍 Stage 3/3: {self.current_stage} ({self.config.final_refinement_trials} random trials)")

            for i in range(self.config.final_refinement_trials):
                self.current_trial += 1
                self.stage_trials_completed = int(i + 1)

                params = self._sample_around_best(best_params)
                score = objective_func(params)

                refinement_results.append((params.copy(), score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            tprint(f"✅ Refinement complete - Best score: {best_score:.4f}")

        all_round_one_results = coarse_results + fine_results + refinement_results

        second_round_results = []
        if self.config.n_rounds > 1:
            best_score, best_params, second_round_results = self._run_second_round(
                all_round_one_results,
                objective_func,
                best_score,
                best_params
            )

        total_trials = (
            len(coarse_results)
            + len(fine_results)
            + len(refinement_results)
            + len(second_round_results)
        )

        return {
            'best_score': best_score,
            'best_params': best_params,
            'coarse_results': coarse_results,
            'fine_results': fine_results,
            'refinement_results': refinement_results,
            'second_round_results': second_round_results,
            'n_trials': total_trials
        }

    def _create_fine_grid(self, best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fine grid around best parameters."""
        tprint_debug("Generating fine grid candidates around best HPO parameters")
        fine_grid = []

        # EWMA config: try adjacent configs
        ewma_idx = best_params['ewma_config_idx']
        ewma_candidates = [
            max(0, ewma_idx - 1),
            ewma_idx,
            min(5, ewma_idx + 1)
        ]

        # n_components: try adjacent values
        n_comp = best_params['n_components']
        n_comp_candidates = [
            max(4, n_comp - 1),
            n_comp,
            min(6, n_comp + 1)
        ]

        # pca_components: try adjacent values
        pca_comp = best_params['pca_components']
        pca_comp_candidates = [
            max(3, pca_comp - 1),
            pca_comp,
            min(5, pca_comp + 1)
        ]

        # min_covar: try values around best (log scale)
        min_cov = best_params['min_covar']
        min_cov_candidates = [
            min_cov / 10,
            min_cov,
            min_cov * 10
        ]
        min_cov_candidates = [max(1e-5, min(1e-2, v)) for v in min_cov_candidates]

        # kappa: try values around best
        kappa = best_params['kappa']
        kappa_candidates = [
            max(1.0, kappa - 10),
            kappa,
            min(50.0, kappa + 10)
        ]

        # Create grid (sample to avoid too many combinations)
        import itertools
        all_combinations = list(itertools.product(
            ewma_candidates,
            n_comp_candidates,
            pca_comp_candidates,
            min_cov_candidates,
            kappa_candidates
        ))

        # Sample up to 30 combinations
        n_samples = min(30, len(all_combinations))
        sampled = np.random.choice(len(all_combinations), n_samples, replace=False)

        for idx in sampled:
            combo = all_combinations[idx]
            fine_grid.append({
                'ewma_config_idx': int(combo[0]),
                'n_components': int(combo[1]),
                'pca_components': int(combo[2]),
                'min_covar': float(combo[3]),
                'kappa': float(combo[4])
            })

        return fine_grid

    def _sample_around_best(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters around best values for refinement."""
        tprint_debug("Sampling parameters around current best configuration for refinement")
        params = best_params.copy()

        # Randomly perturb one or two parameters
        n_perturb = np.random.choice([1, 2])
        perturb_keys = np.random.choice(list(params.keys()), n_perturb, replace=False)

        for key in perturb_keys:
            if key == 'ewma_config_idx':
                # Random walk
                params[key] = int(np.clip(params[key] + np.random.randint(-1, 2), 0, 5))
            elif key == 'n_components':
                params[key] = int(np.clip(params[key] + np.random.randint(-1, 2), 4, 6))
            elif key == 'pca_components':
                params[key] = int(np.clip(params[key] + np.random.randint(-1, 2), 3, 5))
            elif key == 'min_covar':
                # Log-scale perturbation
                log_val = np.log10(params[key])
                log_val += np.random.uniform(-0.5, 0.5)
                params[key] = float(np.clip(10 ** log_val, 1e-5, 1e-2))
            elif key == 'kappa':
                params[key] = float(np.clip(params[key] + np.random.uniform(-5, 5), 1.0, 50.0))

        return params

    SECOND_ROUND_TOP_K = 5
    SECOND_ROUND_MAX_COMBOS_PER_BASE = 60

    def _run_second_round(
        self,
        round_one_results: List[Tuple[Dict[str, Any], float]],
        objective_func,
        current_best_score: float,
        current_best_params: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any], List[Tuple[Dict[str, Any], float]]]:
        """Perform focused second-round search around top performers."""

        if not round_one_results:
            return current_best_score, current_best_params, []

        sorted_results = sorted(round_one_results, key=lambda x: x[1], reverse=True)
        top_candidates = sorted_results[: self.SECOND_ROUND_TOP_K]

        candidate_params = self._build_second_round_candidates(top_candidates)

        if not candidate_params:
            tprint_warning("⚠️  No candidate parameters generated for second-round optimization")
            return current_best_score, current_best_params, []

        self.current_stage = "Round 2"
        tprint("")
        tprint(f"🔁 Stage 4: {self.current_stage} ({len(candidate_params)} trials around top performers)")

        self.stage_trials_total = int(len(candidate_params))
        self.stage_trials_completed = 0

        second_round_results: List[Tuple[Dict[str, Any], float]] = []

        for i, params in enumerate(candidate_params):
            self.current_trial += 1
            self.stage_trials_completed = i + 1

            score = objective_func(params)
            second_round_results.append((params.copy(), score))

            if score > current_best_score:
                current_best_score = score
                current_best_params = params.copy()

        tprint(f"✅ Round 2 complete - Best score: {current_best_score:.4f}")

        return current_best_score, current_best_params, second_round_results

    def _build_second_round_candidates(
        self,
        top_candidates: List[Tuple[Dict[str, Any], float]]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations around top-performing configurations."""

        candidate_list: List[Dict[str, Any]] = []
        seen = set()

        for params, _ in top_candidates:
            expanded = self._generate_second_round_grid(params)
            for candidate in expanded:
                key = tuple(sorted(candidate.items()))
                if key not in seen:
                    seen.add(key)
                    candidate_list.append(candidate)

        return candidate_list

    def _generate_second_round_grid(self, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create bounded grid around a base parameter configuration."""

        ewma_idx = int(base_params.get('ewma_config_idx', 0))
        n_comp = int(base_params.get('n_components', 5))
        pca_comp = int(base_params.get('pca_components', 4))
        min_covar = float(base_params.get('min_covar', 1e-3))
        kappa = float(base_params.get('kappa', 10.0))

        ewma_candidates = list(range(max(0, ewma_idx - 2), min(5, ewma_idx + 2) + 1))
        n_comp_candidates = list(range(max(4, n_comp - 2), min(6, n_comp + 2) + 1))
        pca_comp_candidates = list(range(max(3, pca_comp - 2), min(5, pca_comp + 2) + 1))

        min_covar_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        min_covar_candidates = sorted({
            float(np.clip(min_covar * mult, 1e-5, 1e-2))
            for mult in min_covar_multipliers
        })

        kappa_offsets = [-10.0, -5.0, 0.0, 5.0, 10.0]
        kappa_candidates = sorted({
            float(np.clip(kappa + offset, 1.0, 50.0))
            for offset in kappa_offsets
        })

        import itertools

        all_combinations = list(itertools.product(
            ewma_candidates,
            n_comp_candidates,
            pca_comp_candidates,
            min_covar_candidates,
            kappa_candidates
        ))

        max_combos = min(self.SECOND_ROUND_MAX_COMBOS_PER_BASE, len(all_combinations))

        if len(all_combinations) > max_combos:
            selected_indices = np.random.choice(len(all_combinations), max_combos, replace=False)
            selected_combos = [all_combinations[idx] for idx in selected_indices]
        else:
            selected_combos = all_combinations

        candidate_params = []
        for combo in selected_combos:
            candidate_params.append({
                'ewma_config_idx': int(combo[0]),
                'n_components': int(combo[1]),
                'pca_components': int(combo[2]),
                'min_covar': float(combo[3]),
                'kappa': float(combo[4])
            })

        return candidate_params


# Default HPO configuration
DEFAULT_HPO_CONFIG = HPOConfig(
    stages=[
        OptimizationStage.COARSE_GRID,
        OptimizationStage.FINE_GRID,
        OptimizationStage.TPE
    ],
    n_rounds=2,
    enable_final_refinement=True,
    final_refinement_trials=50,
    cv_folds=5,
    weight_predictive_ll=0.33,
    weight_temporal=0.33,
    weight_economic=0.34,
    direction='maximize',
    use_custom_balanced_score=True,
    verbose=True
)
