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
from typing import Dict, Any, List, Optional, Tuple, Set
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
FORWARD_RETURN_HORIZON = 2
SHARPE_EPS = 1e-9


def _compute_tail_separation_score(
    forward_returns: pd.Series,
    regime_labels: np.ndarray,
    percentile: float = 5.0
) -> Optional[float]:
    """Compute between/within separation for downside-tail (5th percentile) returns."""

    if forward_returns is None or regime_labels is None:
        return None

    min_len = min(len(forward_returns), len(regime_labels))
    if min_len < 5:
        return None

    returns = forward_returns.iloc[:min_len].to_numpy(dtype=float, copy=False)
    labels = np.asarray(regime_labels[:min_len])

    mask = ~np.isnan(returns)
    returns = returns[mask]
    labels = labels[mask]

    if returns.size < 5:
        return None

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    if unique_labels.size < 2:
        return None

    regime_tail_values = []
    for lbl in unique_labels:
        subset = returns[labels == lbl]
        if subset.size < 3:
            continue
        tail_val = float(np.percentile(subset, percentile))
        regime_tail_values.append(tail_val)

    if len(regime_tail_values) < 2:
        return None

    between = float(np.std(regime_tail_values))
    within = float(np.std(returns)) + CV_RATIO_EPS

    if within <= CV_RATIO_EPS:
        return None

    return between / within


def _compute_cv_ratio_for_horizon(
    forward_returns: pd.Series,
    regime_labels: np.ndarray
) -> Optional[float]:
    """Compute between/within CV ratio for a given forward-return horizon."""

    if forward_returns is None or regime_labels is None:
        return None

    min_len = min(len(forward_returns), len(regime_labels))
    if min_len < 5:
        return None

    returns = forward_returns.iloc[:min_len].to_numpy(dtype=float, copy=False)
    labels = np.asarray(regime_labels[:min_len])

    mask = ~np.isnan(returns)
    returns = returns[mask]
    labels = labels[mask]

    if returns.size < 5:
        return None

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    if unique_labels.size < 2:
        return None

    regime_means = []
    regime_stds = []
    for lbl in unique_labels:
        subset = returns[labels == lbl]
        if subset.size < 3:
            continue
        regime_means.append(np.mean(subset))
        regime_stds.append(np.std(subset))

    if len(regime_means) < 2 or not regime_stds:
        return None

    between = float(np.std(regime_means))
    within = float(np.mean(regime_stds)) + CV_RATIO_EPS

    if within <= CV_RATIO_EPS:
        return None

    return between / within


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
    weight_between_within_cv: float = 0.40
    weight_temporal: float = 0.20
    weight_economic: float = 0.40

    # Optimization settings
    direction: str = 'maximize'
    use_custom_balanced_score: bool = True
    verbose: bool = True

    # Early stopping heuristics
    enable_early_stopping: bool = True
    early_stop_min_score: float = 0.05
    early_stop_min_quality_score: float = 0.1
    early_stop_min_temporal_smoothness: float = 0.1
    early_stop_patience: int = 5

    # Stage seeding behaviour
    enable_stage_seeding: bool = True
    seed_pool_top_k: int = 3

    def __post_init__(self):
        tprint_debug("Initializing HPOConfig dataclass")
        if self.stages is None:
            self.stages = [
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ]

        # Validate weights sum to 1
        total_weight = self.weight_between_within_cv + self.weight_temporal + self.weight_economic
        if not np.isclose(total_weight, 1.0):
            tprint_error(
                f"⚠️  Invalid objective weights detected (sum={total_weight:.4f}); expected 1.0"
            )
            raise ValueError(f"Objective weights must sum to 1.0, got {total_weight}")

        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be >= 1")

        if self.seed_pool_top_k < 1:
            raise ValueError("seed_pool_top_k must be >= 1")


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
                        "high": 7,
                        "step": 1
                    },
                    "pca_components": {
                        "type": "int",
                        "low": 5,
                        "high": 10,
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

        def objective(params: Dict[str, Any]) -> Tuple[float, Optional[Any]]:
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
                pca_components = int(params.get('pca_components', 7))
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
                    return -1e6, None

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
                hmm_model.fit(
                    features_pca.values,
                    ewma_config_name=ewma_config.name,
                    pca_components=pca_components
                )

                # Predict regime labels
                regime_labels = hmm_model.predict(features_pca.values)

                # Get transition matrix
                transition_matrix = hmm_model.get_transition_matrix()

                # Calculate forward returns (2-bar default horizon)
                forward_returns = (
                    market_data['close'].pct_change(FORWARD_RETURN_HORIZON)
                    .shift(-FORWARD_RETURN_HORIZON)
                )
                forward_returns = forward_returns.loc[features_pca.index]

                # Assess quality with fast mode enabled (skip expensive O(n²) calculations)
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
                    temporal_sensitivity_mode="standard",
                    fast_mode=True  # Enable fast mode to skip silhouette/DBI calculations
                )

                # Calculate objective score using available metrics
                between_cv = getattr(metrics, 'between_regime_cv', 0.0)
                within_cv = getattr(metrics, 'within_regime_cv', 0.0)
                cv_ratio = between_cv / (within_cv + CV_RATIO_EPS) if within_cv is not None else 0.0
                normalized_cv_ratio = cv_ratio / (cv_ratio + 1.0) if cv_ratio > 0 else 0.0

                silhouette_raw = getattr(metrics, 'silhouette_score', 0.0)
                silhouette_norm = float(np.clip((silhouette_raw + 1.0) / 2.0, 0.0, 1.0))

                stat_score = float(np.clip((normalized_cv_ratio + silhouette_norm) / 2.0, 0.0, 1.0))
                score_statistical = stat_score * self.config.weight_between_within_cv

                temporal_smoothness = getattr(metrics, 'temporal_smoothness', 0.0)
                temporal_score = getattr(metrics, 'comprehensive_temporal_score', temporal_smoothness)
                score_temporal = (
                    temporal_smoothness * 0.5 + temporal_score * 0.5
                ) * self.config.weight_temporal

                economic_components: List[float] = []

                economic_cv_ratio = 0.0
                if hasattr(metrics, 'economic_cv_metrics') and metrics.economic_cv_metrics:
                    economic_cv_ratio = metrics.economic_cv_metrics.get('economic_cv_ratio_mean_return', 0.0) or 0.0
                    if economic_cv_ratio > 0.0:
                        economic_components.append(
                            float(np.clip(economic_cv_ratio / (economic_cv_ratio + 1.0), 0.0, 1.0))
                        )

                if hasattr(metrics, 'economic_validation') and metrics.economic_validation:
                    mean_returns = [
                        regime_data.get('mean_return')
                        for regime_data in metrics.economic_validation.values()
                        if isinstance(regime_data, dict)
                    ]
                    volatilities = [
                        regime_data.get('volatility')
                        for regime_data in metrics.economic_validation.values()
                        if isinstance(regime_data, dict)
                    ]
                    mean_returns = [m for m in mean_returns if m is not None]
                    volatilities = [v for v in volatilities if v is not None]

                    if mean_returns and volatilities:
                        avg_mean = float(np.mean(mean_returns))
                        avg_vol = float(np.mean(volatilities))
                        if avg_vol > SHARPE_EPS:
                            normalized_sharpe = float(np.clip((avg_mean / avg_vol + 2.0) / 6.0, 0.0, 1.0))
                            economic_components.append(normalized_sharpe)

                extended_cv_scores: List[float] = []
                horizon_configs = [2]

                close_series = pd.Series(market_data['close'], copy=False)
                volume_series = pd.Series(market_data['volume'], copy=False)

                base_series: Dict[str, pd.Series] = {
                    'close_price': close_series,
                    'volume_level': volume_series,
                }
                if {'high', 'low', 'close'}.issubset(market_data.columns):
                    range_ratio = (market_data['high'] - market_data['low']).divide(
                        market_data['close'].replace(0.0, np.nan)
                    )
                    range_ratio = pd.Series(range_ratio, copy=False)
                    base_series['intraday_range'] = range_ratio

                for horizon in horizon_configs:
                    for series_name, series in base_series.items():
                        if series is None:
                            continue
                        series_forward = series.pct_change(horizon).shift(-horizon)
                        series_forward = series_forward.loc[features_pca.index]
                        horizon_ratio = _compute_cv_ratio_for_horizon(series_forward, regime_labels)
                        if horizon_ratio is not None and horizon_ratio > 0:
                            normalized_ratio = float(
                                np.clip(horizon_ratio / (horizon_ratio + 1.0), 0.0, 1.0)
                            )
                            extended_cv_scores.append(normalized_ratio)
                            if hasattr(metrics, 'economic_cv_metrics') and isinstance(metrics.economic_cv_metrics, dict):
                                metrics.economic_cv_metrics[
                                    f'{series_name}_cv_ratio_h{horizon}'
                                ] = normalized_ratio

                if extended_cv_scores:
                    economic_components.append(float(np.mean(extended_cv_scores)))

                # Downside-tail metrics (5th percentile 2-bar return) to reward tail separation
                tail_components: List[float] = []
                tail_horizon = 2
                for series_name, series in base_series.items():
                    if series is None:
                        continue
                    series_forward = series.pct_change(tail_horizon).shift(-tail_horizon)
                    series_forward = series_forward.loc[features_pca.index]
                    tail_ratio = _compute_tail_separation_score(series_forward, regime_labels)
                    if tail_ratio is not None and tail_ratio > 0:
                        tail_components.append(tail_ratio)
                        if hasattr(metrics, 'economic_cv_metrics') and isinstance(metrics.economic_cv_metrics, dict):
                            metrics.economic_cv_metrics[
                                f'{series_name}_tail_separation_h{tail_horizon}'
                            ] = tail_ratio

                if tail_components:
                    economic_components.append(float(np.mean(tail_components)))

                if economic_components:
                    economic_signal = float(np.mean(economic_components))
                else:
                    economic_signal = 0.0

                score_economic = economic_signal * self.config.weight_economic

                persistence_penalty = 0.0
                if transition_matrix is not None:
                    diag_mean = float(np.mean(np.diag(transition_matrix)))
                    persistence_penalty += max(0.0, diag_mean - 0.8) * 2.0
                    diag_variance = float(np.var(np.diag(transition_matrix)))
                    persistence_penalty += diag_variance * 0.5
                if metrics.regime_persistence is not None:
                    normalized_persistence = min(1.0, metrics.regime_persistence / 30.0)
                    persistence_penalty += normalized_persistence * 0.2

                
                objective_score = (
                    score_statistical
                    + score_temporal
                    + score_economic
                    - persistence_penalty * self.config.weight_temporal
                )

                return objective_score, metrics

            except Exception as e:
                self.logger.warning(f"Objective function failed: {e}")
                tprint_warning(f"  ⚠️  Trial {self.current_trial} failed: {str(e)[:80]}")
                return -1e6, None

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
            f"pca={params.get('pca_components', 6)}, "  # Broadened PCA search range
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
        """Custom hierarchical optimization with early stopping, seeding, and successive halving."""

        tprint_info("Running custom hierarchical optimization with successive halving")

        coarse_grid = {
            'ewma_config_idx': [0, 2, 4],
            'n_components': [4, 5, 6, 7],
            'pca_components': [5, 6, 7, 8, 9, 10],
            'min_covar': [1e-5, 1e-4, 1e-3, 1e-2],  # Reduced from 5 to 4 values (removed 2e-2)
            'kappa': [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]  # Reduced from 17 to 15 values (removed 25.0, 30.0)
        }

        best_score = -np.inf
        best_params: Optional[Dict[str, Any]] = None
        coarse_results: List[Dict[str, Any]] = []
        fine_results: List[Dict[str, Any]] = []
        refinement_results: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Tuple[str, Any], ...]] = set()

        # Stage 1 – coarse sampling with successive halving
        n_initial_candidates = min(60, int(np.prod([len(v) for v in coarse_grid.values()])))
        self.current_stage = "Coarse Grid with Successive Halving"
        self.stage_trials_total = n_initial_candidates
        self.stage_trials_completed = 0

        tprint("")
        tprint(f"🔍 Stage 1/3: {self.current_stage} ({n_initial_candidates} initial candidates)")

        # Generate initial candidates
        initial_candidates = []
        for _ in range(n_initial_candidates):
            params = {key: np.random.choice(values) for key, values in coarse_grid.items()}
            signature = self._params_signature(params)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                initial_candidates.append(params)

        # Apply successive halving
        successive_halving_results = self._apply_successive_halving(
            initial_candidates, objective_func, n_rungs=3, reduction_factor=3
        )
        
        coarse_results = successive_halving_results['all_results']
        best_score = successive_halving_results['best_score']
        best_params = successive_halving_results['best_params']

        # Stage 2 – fine grid around best region
        self.current_stage = "Fine Grid"
        fine_candidates = self._prepare_fine_candidates(
            coarse_results,
            self._create_fine_grid(best_params or {}),
            best_params
        )
        self.stage_trials_total = len(fine_candidates)
        self.stage_trials_completed = 0

        tprint("")
        tprint(f"🔍 Stage 2/3: {self.current_stage} ({len(fine_candidates)} trials around best region)")

        fine_poor_counter = 0

        for params in fine_candidates:
            signature = self._params_signature(params)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            self.current_trial += 1
            self.stage_trials_completed += 1

            score, metrics = objective_func(params)
            metrics_dict = metrics.to_dict() if metrics is not None and hasattr(metrics, "to_dict") else None
            fine_results.append(
                {
                    'params': params.copy(),
                    'score': score,
                    'quality_metrics': metrics_dict,
                    'trial_number': self.current_trial
                }
            )

            if score is not None and score > best_score:
                best_score = score
                best_params = params.copy()

            fine_poor_counter = self._update_poor_counter(score, metrics, fine_poor_counter)
            if self._should_early_stop(fine_poor_counter, "Fine Grid"):
                break

        tprint(f"✅ Fine grid complete - Best score: {best_score:.4f}")

        # Stage 3 – focused refinement via seeding + perturbations
        if self.config.enable_final_refinement and best_params is not None:
            refinement_candidates = self._prepare_refinement_candidates(
                best_params,
                coarse_results,
                fine_results,
                seen_signatures
            )

            self.current_stage = "Refinement"
            self.stage_trials_total = len(refinement_candidates)
            self.stage_trials_completed = 0

            tprint("")
            tprint(f"🔍 Stage 3/3: {self.current_stage} ({len(refinement_candidates)} focused trials)")

            refinement_poor_counter = 0

            for params in refinement_candidates:
                signature = self._params_signature(params)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                self.current_trial += 1
                self.stage_trials_completed += 1

                score, metrics = objective_func(params)
                metrics_dict = metrics.to_dict() if metrics is not None and hasattr(metrics, "to_dict") else None
                refinement_results.append(
                    {
                        'params': params.copy(),
                        'score': score,
                        'quality_metrics': metrics_dict,
                        'trial_number': self.current_trial
                    }
                )

                if score is not None and score > best_score:
                    best_score = score
                    best_params = params.copy()

                refinement_poor_counter = self._update_poor_counter(score, metrics, refinement_poor_counter)
                if self._should_early_stop(refinement_poor_counter, "Refinement"):
                    break

            tprint(f"✅ Refinement complete - Best score: {best_score:.4f}")

        all_round_one_results = coarse_results + fine_results + refinement_results

        second_round_results: List[Dict[str, Any]] = []
        if self.config.n_rounds > 1:
            best_score, best_params, second_round_results = self._run_second_round(
                all_round_one_results,
                objective_func,
                best_score,
                best_params or {}
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

    def _params_signature(self, params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """Create a hashable signature for parameter dictionaries."""

        signature: List[Tuple[str, Any]] = []
        for key, value in sorted(params.items()):
            if isinstance(value, float):
                signature.append((key, float(f"{value:.10f}")))
            else:
                signature.append((key, value))
        return tuple(signature)

    def _update_poor_counter(
        self,
        score: Optional[float],
        metrics: Optional[Any],
        current_counter: int
    ) -> int:
        """Advance the consecutive low-quality counter based on heuristics."""

        if not self.config.enable_early_stopping:
            return 0

        if score is None or score < self.config.early_stop_min_score:
            return current_counter + 1

        if metrics is None:
            return current_counter + 1

        quality_score = getattr(metrics, 'quality_score', None)
        if quality_score is not None and quality_score < self.config.early_stop_min_quality_score:
            return current_counter + 1

        temporal_smoothness = getattr(metrics, 'temporal_smoothness', None)
        if (
            temporal_smoothness is not None
            and temporal_smoothness < self.config.early_stop_min_temporal_smoothness
        ):
            return current_counter + 1

        return 0

    def _should_early_stop(self, poor_counter: int, stage: str) -> bool:
        """Return True when the current stage should halt early."""

        if not self.config.enable_early_stopping:
            return False

        if poor_counter >= self.config.early_stop_patience:
            tprint_warning(
                f"  ⚠️  Early stopping triggered in {stage} after {poor_counter} consecutive low-quality trials"
            )
            return True

        return False

    def _extract_seed_params(
        self,
        results: List[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Collect the top-performing parameter sets to reuse as seeds."""

        if not self.config.enable_stage_seeding:
            return [fallback.copy()] if fallback is not None else []

        valid_results = [r for r in results if r.get('score') is not None and r.get('params')]
        sorted_results = sorted(valid_results, key=lambda entry: entry['score'], reverse=True)

        seeds: List[Dict[str, Any]] = []
        for entry in sorted_results[: self.config.seed_pool_top_k]:
            params = entry.get('params')
            if not params:
                continue
            seeds.append(params.copy())

        if fallback is not None and (not seeds or fallback not in seeds):
            seeds.insert(0, fallback.copy())

        deduped: List[Dict[str, Any]] = []
        seen: Set[Tuple[Tuple[str, Any], ...]] = set()
        for params in seeds:
            signature = self._params_signature(params)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(params)

        return deduped

    def _apply_successive_halving(
        self,
        candidates: List[Dict[str, Any]],
        objective_func,
        n_rungs: int = 3,
        reduction_factor: int = 3
    ) -> Dict[str, Any]:
        """
        Apply successive halving algorithm to eliminate poor performing configurations early.
        
        Args:
            candidates: List of parameter configurations to evaluate
            objective_func: Objective function to evaluate configurations
            n_rungs: Number of successive halving rounds
            reduction_factor: Factor to reduce candidates by each round
            
        Returns:
            Dictionary with results including best_score, best_params, and all_results
        """
        tprint_info(f"🔪 Applying successive halving: {len(candidates)} candidates, {n_rungs} rungs, reduction factor {reduction_factor}")
        
        current_candidates = candidates.copy()
        all_results: List[Dict[str, Any]] = []
        best_score = -np.inf
        best_params: Optional[Dict[str, Any]] = None
        
        for rung in range(n_rungs):
            n_evaluations = max(1, len(current_candidates) // (reduction_factor ** rung))
            n_survivors = max(1, len(current_candidates) // (reduction_factor ** (rung + 1)))
            
            tprint_info(f"  Rung {rung + 1}/{n_rungs}: Evaluating {n_evaluations} candidates, promoting {n_survivors}")
            
            # Evaluate current candidates
            rung_results = []
            for params in current_candidates[:n_evaluations]:
                self.current_trial += 1
                self.stage_trials_completed += 1
                
                score, metrics = objective_func(params)
                metrics_dict = metrics.to_dict() if metrics is not None and hasattr(metrics, "to_dict") else None
                
                result = {
                    'params': params.copy(),
                    'score': score,
                    'quality_metrics': metrics_dict,
                    'trial_number': self.current_trial,
                    'rung': rung + 1
                }
                
                rung_results.append(result)
                all_results.append(result)
                
                if score is not None and score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            # Sort by score and select survivors
            rung_results.sort(key=lambda x: x['score'] if x['score'] is not None else -np.inf, reverse=True)
            current_candidates = [result['params'] for result in rung_results[:n_survivors]]
            
            # Log progress
            if rung_results:
                best_rung_score = rung_results[0]['score']
                worst_rung_score = rung_results[-1]['score'] if rung_results[-1]['score'] is not None else -np.inf
                tprint_info(f"    Best score this rung: {best_rung_score:.4f}, Worst: {worst_rung_score:.4f}")
            
            # Early termination if only one candidate left
            if len(current_candidates) <= 1:
                tprint_info(f"  Early termination: only {len(current_candidates)} candidate(s) remaining")
                break
        
        tprint_info(f"✅ Successive halving complete: best_score={best_score:.4f}")
        
        return {
            'all_results': all_results,
            'best_score': best_score,
            'best_params': best_params,
            'final_candidates': current_candidates
        }

    def _prepare_fine_candidates(
        self,
        coarse_results: List[Dict[str, Any]],
        fine_grid: List[Dict[str, Any]],
        best_params: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Blend seed parameters with enumerated fine-grid combinations."""

        candidates: List[Dict[str, Any]] = []
        seen: Set[Tuple[Tuple[str, Any], ...]] = set()

        for params in self._extract_seed_params(coarse_results, fallback=best_params) + fine_grid:
            signature = self._params_signature(params)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(params.copy())

        return candidates

    def _prepare_refinement_candidates(
        self,
        best_params: Optional[Dict[str, Any]],
        coarse_results: List[Dict[str, Any]],
        fine_results: List[Dict[str, Any]],
        seen_signatures: Set[Tuple[Tuple[str, Any], ...]]
    ) -> List[Dict[str, Any]]:
        """Generate refinement candidates using seeds and perturbations."""

        seeds = self._extract_seed_params(fine_results if fine_results else coarse_results, fallback=best_params)

        candidates: List[Dict[str, Any]] = []
        stage_seen: Set[Tuple[Tuple[str, Any], ...]] = set(seen_signatures)
        target_trials = max(0, int(self.config.final_refinement_trials))

        for seed in seeds:
            if target_trials and len(candidates) >= target_trials:
                break
            signature = self._params_signature(seed)
            if signature in stage_seen:
                continue
            stage_seen.add(signature)
            candidates.append(seed.copy())

        attempts = 0
        max_attempts = max(target_trials * 5, 20) if target_trials else 20

        while (not target_trials or len(candidates) < target_trials) and attempts < max_attempts:
            attempts += 1
            sampled = self._sample_around_best(best_params or {})
            if not sampled:
                break

            signature = self._params_signature(sampled)
            if signature in stage_seen:
                continue
            stage_seen.add(signature)
            candidates.append(sampled)

        if target_trials:
            return candidates[:target_trials]
        return candidates

    def _create_fine_grid(self, best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fine grid around best parameters."""
        tprint_debug("Generating fine grid candidates around best HPO parameters")
        fine_grid = []
        import itertools

        # Generate candidates around best parameters
        best_ewma = best_params.get('ewma_config_idx', 0)
        best_n_comp = best_params.get('n_components', 5)
        best_pca_comp = best_params.get('pca_components', 7)
        best_min_cov = best_params.get('min_covar', 1e-3)
        best_kappa = best_params.get('kappa', 10.0)

        # Create fine-tuned ranges around best values
        ewma_candidates = [max(0, best_ewma - 1), best_ewma, min(5, best_ewma + 1)]
        n_comp_candidates = [max(4, best_n_comp - 1), best_n_comp, min(7, best_n_comp + 1)]
        pca_comp_candidates = [max(5, best_pca_comp - 1), best_pca_comp, min(10, best_pca_comp + 1)]
        min_cov_candidates = [best_min_cov * 0.5, best_min_cov, best_min_cov * 2.0]
        kappa_candidates = [best_kappa * 0.5, best_kappa, best_kappa * 1.5]

        all_combinations = list(itertools.product(
            ewma_candidates,
            n_comp_candidates,
            pca_comp_candidates,
            min_cov_candidates,
            kappa_candidates
        ))

        fine_grid: List[Dict[str, Any]] = []
        for combo in all_combinations:
            fine_grid.append({
                'ewma_config_idx': int(combo[0]),
                'n_components': int(combo[1]),
                'pca_components': int(combo[2]),
                'min_covar': float(combo[3]),
                'kappa': float(combo[4])
            })

        return fine_grid

    def _prepare_refinement_candidates(
        self,
        best_params: Optional[Dict[str, Any]],
        coarse_results: List[Dict[str, Any]],
        fine_results: List[Dict[str, Any]],
        seen_signatures: Set[Tuple[Tuple[str, Any], ...]]
    ) -> List[Dict[str, Any]]:
        """Seed refinement stage with perturbations around top configurations."""

        if not best_params:
            return []

        seed_pool: List[Dict[str, Any]] = [best_params]

        if self.config.enable_stage_seeding:
            def _top_k(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                scored = [r for r in results if r.get('params') and r.get('score') is not None]
                return sorted(scored, key=lambda r: r['score'], reverse=True)[: self.config.seed_pool_top_k]

            seed_pool.extend([r['params'] for r in _top_k(coarse_results)])
            seed_pool.extend([r['params'] for r in _top_k(fine_results)])

        unique_signatures = set()
        candidates: List[Dict[str, Any]] = []
        max_candidates = max(1, self.config.final_refinement_trials)
        attempts = 0

        if not seed_pool:
            return []

        while len(candidates) < max_candidates and attempts < max_candidates * 5:
            base = seed_pool[attempts % len(seed_pool)]
            params = self._sample_around_best(base)
            attempts += 1

            if not params:
                continue

            signature = self._params_signature(params)
            if signature in seen_signatures or signature in unique_signatures:
                continue

            unique_signatures.add(signature)
            candidates.append(params)

        return candidates

    def _sample_around_best(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters around best values for refinement."""

        tprint_debug("Sampling parameters around current best configuration for refinement")

        if not best_params:
            return {}

        params = best_params.copy()

        n_perturb = np.random.choice([1, 2])
        perturb_keys = np.random.choice(list(params.keys()), n_perturb, replace=False)

        for key in perturb_keys:
            if key == 'ewma_config_idx':
                params[key] = int(np.clip(params[key] + np.random.randint(-1, 2), 0, 5))
            elif key == 'n_components':
                params[key] = int(np.clip(params[key] + np.random.randint(-1, 2), 4, 7))
            elif key == 'pca_components':
                params[key] = int(np.clip(params[key] + np.random.randint(-1, 2), 5, 10))
            elif key == 'min_covar':
                log_val = np.log10(params[key])
                log_val += np.random.uniform(-0.5, 0.5)
                params[key] = float(np.clip(10 ** log_val, 1e-5, 1e-2))  # Cap at 0.01 (reduced from 0.02)
            elif key == 'kappa':
                params[key] = float(np.clip(params[key] + np.random.uniform(-5, 5), 0.25, 20.0))  # Cap at 20.0 (reduced from 30.0)

        return params

    SECOND_ROUND_TOP_K = 5
    SECOND_ROUND_MAX_COMBOS_PER_BASE = 60

    def _run_second_round(
        self,
        round_one_results: List[Dict[str, Any]],
        objective_func,
        current_best_score: float,
        current_best_params: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Perform focused second-round search around top performers."""

        if not round_one_results:
            return current_best_score, current_best_params, []

        sorted_results = sorted(
            [r for r in round_one_results if r.get('score') is not None],
            key=lambda x: x['score'],
            reverse=True
        )
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

        second_round_results: List[Dict[str, Any]] = []

        for i, params in enumerate(candidate_params):
            self.current_trial += 1
            self.stage_trials_completed = i + 1

            score, metrics = objective_func(params)
            second_round_results.append(
                {
                    'params': params.copy(),
                    'score': score,
                    'quality_metrics': metrics.to_dict() if metrics is not None else None,
                    'trial_number': self.current_trial
                }
            )

            if score > current_best_score:
                current_best_score = score
                current_best_params = params.copy()

        tprint(f"✅ Round 2 complete - Best score: {current_best_score:.4f}")

        return current_best_score, current_best_params, second_round_results

    def _build_second_round_candidates(
        self,
        top_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations around top-performing configurations."""

        candidate_list: List[Dict[str, Any]] = []
        seen = set()

        for candidate in top_candidates:
            params = candidate.get('params', {})
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
        pca_comp = int(base_params.get('pca_components', 7))
        min_covar = float(base_params.get('min_covar', 1e-3))
        kappa = float(base_params.get('kappa', 10.0))

        ewma_candidates = list(range(max(0, ewma_idx - 2), min(5, ewma_idx + 2) + 1))
        n_comp_candidates = list(range(max(4, n_comp - 2), min(6, n_comp + 2) + 1))
        pca_comp_candidates = list(range(max(5, pca_comp - 2), min(10, pca_comp + 2) + 1))

        min_covar_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        min_covar_candidates = sorted({
            float(np.clip(min_covar * mult, 1e-5, 1e-2))  # Cap at 0.01 (reduced from 0.02)
            for mult in min_covar_multipliers
        })

        kappa_offsets = [-5.0, -2.5, 0.0, 2.5, 5.0]  # Refine kappa grid
        kappa_candidates = sorted({
            float(np.clip(kappa + offset, 0.25, 20.0))  # Cap at 20.0 (reduced from 30.0)
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

    def _params_signature(self, params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """Create a signature for parameters to track duplicates."""

        signature_items: List[Tuple[str, Any]] = []
        for key, value in sorted(params.items()):
            if isinstance(value, float):
                signature_items.append((key, float(f"{value:.10f}")))
            else:
                signature_items.append((key, value))
        return tuple(signature_items)

    def _update_poor_counter(
        self,
        score: Optional[float],
        metrics: Optional[Any],
        current_counter: int
    ) -> int:
        """Advance the consecutive low-quality counter using heuristics."""

        if not self.config.enable_early_stopping:
            return 0

        if score is None or score < self.config.early_stop_min_score:
            return current_counter + 1

        if metrics is None:
            return current_counter + 1

        quality = getattr(metrics, 'quality_score', 0.0)
        temporal = getattr(metrics, 'temporal_smoothness', 0.0)

        if quality < self.config.early_stop_min_quality_score:
            return current_counter + 1

        if temporal < self.config.early_stop_min_temporal_smoothness:
            return current_counter + 1

        return 0

    def _should_early_stop(self, poor_counter: int, stage: str) -> bool:
        """Determine if early stopping should be triggered."""

        if not self.config.enable_early_stopping:
            return False

        if poor_counter >= self.config.early_stop_patience:
            tprint_warning(
                f"⚠️  Early stopping in {stage} due to {poor_counter} consecutive poor trials"
            )
            return True

        return False


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
    weight_between_within_cv=0.40,
    weight_temporal=0.20,
    weight_economic=0.40,
    direction='maximize',
    use_custom_balanced_score=True,
    verbose=True
)
