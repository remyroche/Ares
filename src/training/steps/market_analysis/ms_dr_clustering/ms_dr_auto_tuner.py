"""
MS-DR Clustering Auto-Tuner

Automatic hyperparameter optimization for MS-DR clustering using:
1. Coarse Grid Search
2. Fine Grid Search around best results
3. TPE (Tree-structured Parzen Estimator) for final optimization

Optimizes the composite quality score from cluster_quality_assessor.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_structured, tprint_timer
)

# Import optimization utilities
from src.utils.ml_common.optimization.auto_tuner import AutoTuner, DatasetCharacteristics
from src.utils.ml_common.optimization.grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig
)

# Import MS-DR components
from .ms_dr_clusterer import MSDRClusterer, MSDRConfig, MSDRResult
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score
)

# Import hierarchical optimization
try:
    from .hierarchical_hpo_extension import (
        MSDRHierarchicalOptimizer,
        create_msdr_parameter_groups,
        create_msdr_optimization_stages
    )
    HIERARCHICAL_HPO_AVAILABLE = True
except ImportError:
    HIERARCHICAL_HPO_AVAILABLE = False
    tprint_debug("Hierarchical HPO extension not available")

logger = logging.getLogger(__name__)


@dataclass
class MSDRTuningConfig:
    """Configuration for MS-DR auto-tuning."""
    # Trial budget
    n_trials: int = 100
    coarse_grid_trials: int = 30
    fine_grid_trials: int = 30
    tpe_trials: int = 40
    
    # Grid granularity
    coarse_grid_points: int = 3
    fine_grid_points: int = 5
    
    # Optimization
    direction: str = 'maximize'  # Maximize composite quality score
    timeout_minutes: float = 60.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    # Hierarchical optimization
    use_hierarchical: bool = False  # Use hierarchical parameter optimization
    n_trials_per_group: int = 30  # Trials per parameter group in hierarchical mode
    
    # Random seed
    random_state: int = 42


class MSDRAutoTuner:
    """
    Automatic hyperparameter tuner for MS-DR clustering.
    
    This tuner optimizes MS-DR clustering parameters to maximize the
    composite quality score from cluster_quality_assessor.py using a
    staged optimization strategy:
    
    Stage 1: Coarse Grid Search (broad exploration)
    Stage 2: Fine Grid Search around best results (local refinement)
    Stage 3: TPE Optimization (final optimization)
    
    Example:
        >>> from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_auto_tuner import (
        ...     MSDRAutoTuner
        ... )
        >>> 
        >>> # Initialize auto-tuner
        >>> tuner = MSDRAutoTuner()
        >>> 
        >>> # Auto-tune on your data
        >>> result = tuner.auto_tune(
        ...     data=market_data,
        ...     n_trials=100,
        ...     timeout_minutes=60.0
        ... )
        >>> 
        >>> # Get best parameters
        >>> best_params = result['best_params']
        >>> best_score = result['best_score']
        >>> 
        >>> # Use best parameters for clustering
        >>> clusterer = MSDRClusterer(MSDRConfig(**best_params))
        >>> ms_result = clusterer.fit_predict(market_data.values)
    """
    
    def __init__(self, tuning_config: Optional[MSDRTuningConfig] = None):
        """
        Initialize MS-DR auto-tuner.
        
        Args:
            tuning_config: Configuration for tuning process
        """
        self.tuning_config = tuning_config or MSDRTuningConfig()
        self.logger = logger
        self.quality_assessor = create_cluster_quality_assessor()
        
        # Optimization history
        self.trial_history: List[Dict[str, Any]] = []
        self.best_result = None
        self.best_score = float('-inf')
        self.best_params = None
        
        tprint_info("🎯 Initialized MS-DR Auto-Tuner")
    
    def get_search_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Define search space for MS-DR hyperparameters.
        
        Returns:
            Search space dictionary compatible with optimization utilities
        """
        search_space = {
            # Number of regimes
            'n_regimes': {
                'type': 'int',
                'low': 3,
                'high': 12
            },
            
            # Model order (for autoregression)
            'order': {
                'type': 'int',
                'low': 1,
                'high': 5
            },
            
            # Switching variance
            'switching_variance': {
                'type': 'categorical',
                'choices': [True, False]
            },
            
            # Model type
            'model_type': {
                'type': 'categorical',
                'choices': ['autoregression', 'regression']
            },
            
            # PCA components
            'pca_components': {
                'type': 'int',
                'low': 5,
                'high': 20
            },
            
            # PCA variance threshold
            'pca_variance_threshold': {
                'type': 'float',
                'low': 0.85,
                'high': 0.99
            }
        }
        
        return search_space
    
    def _create_ms_dr_config(self, params: Dict[str, Any]) -> MSDRConfig:
        """Create MSDRConfig from parameters."""
        return MSDRConfig(
            n_regimes=params.get('n_regimes', 5),
            model_type=params.get('model_type', 'autoregression'),
            order=params.get('order', 1),
            switching_variance=params.get('switching_variance', True),
            enable_pca=True,
            pca_components=params.get('pca_components', 10),
            pca_variance_threshold=params.get('pca_variance_threshold', 0.95),
            auto_select_regimes=False,  # Fixed n_regimes during tuning
            random_state=self.tuning_config.random_state
        )
    
    def _evaluate_params(self, params: Dict[str, Any], data: np.ndarray) -> float:
        """
        Evaluate a set of parameters by running MS-DR clustering
        and computing the composite quality score.
        
        Args:
            params: Parameters to evaluate
            data: Input data
            
        Returns:
            Composite quality score (higher is better)
        """
        try:
            # Create config from parameters
            config = self._create_ms_dr_config(params)
            
            # Run MS-DR clustering
            clusterer = MSDRClusterer(config)
            result = clusterer.fit_predict(data)
            
            if not result.success:
                tprint_warning(f"⚠️ MS-DR clustering failed: {result.error_message}")
                return float('-inf')
            
            # Assess quality using cluster_quality_assessor
            if isinstance(data, np.ndarray):
                feature_df = pd.DataFrame(data)
            else:
                feature_df = data
            
            quality_metrics = self.quality_assessor.assess_quality(
                regime_labels=result.cluster_labels,
                feature_data=feature_df,
                forward_returns=None,
                timestamps=None,
                min_regime_size=10
            )
            
            # Return composite quality score
            composite_score = quality_metrics.quality_score if quality_metrics.quality_score is not None else 0.0
            
            # Store trial result
            self.trial_history.append({
                'params': params,
                'composite_score': composite_score,
                'n_clusters': result.n_clusters,
                'silhouette_score': quality_metrics.silhouette_score,
                'davies_bouldin_score': quality_metrics.davies_bouldin_score,
                'balance_score': quality_metrics.balance_score
            })
            
            # Update best result
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_params = params
                self.best_result = {
                    'params': params,
                    'score': composite_score,
                    'result': result,
                    'quality_metrics': quality_metrics
                }
                
                tprint_success(f"✅ New best score: {composite_score:.4f} (n_regimes={params['n_regimes']})")
            
            return composite_score
            
        except Exception as e:
            tprint_error(f"❌ Error evaluating params: {e}")
            import traceback
            tprint_debug(traceback.format_exc())
            return float('-inf')
    
    def auto_tune(
        self,
        data: pd.DataFrame,
        n_trials: Optional[int] = None,
        timeout_minutes: Optional[float] = None,
        enable_staged_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Automatically tune MS-DR clustering hyperparameters.
        
        Args:
            data: Market data DataFrame
            n_trials: Total number of trials (uses config default if None)
            timeout_minutes: Timeout in minutes (uses config default if None)
            enable_staged_optimization: Use coarse -> fine -> TPE strategy
            
        Returns:
            Dictionary with tuning results:
                - best_params: Best hyperparameters found
                - best_score: Best composite quality score achieved
                - best_result: Full clustering result with best parameters
                - trial_history: History of all trials
                - optimization_summary: Summary statistics
        """
        tprint_info("🚀 Starting MS-DR Auto-Tuning")
        
        # Override config if specified
        if n_trials is not None:
            self.tuning_config.n_trials = n_trials
        if timeout_minutes is not None:
            self.tuning_config.timeout_minutes = timeout_minutes
        
        # Log configuration
        tprint_structured({
            'n_trials': self.tuning_config.n_trials,
            'timeout_minutes': self.tuning_config.timeout_minutes,
            'enable_staged_optimization': enable_staged_optimization,
            'data_shape': data.shape
        }, level="INFO")
        
        # Convert data to numpy if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Reset history
        self.trial_history = []
        self.best_score = float('-inf')
        self.best_params = None
        self.best_result = None
        
        search_space = self.get_search_space()
        
        if enable_staged_optimization:
            # Stage 1: Coarse Grid Search
            tprint_info("📊 Stage 1: Coarse Grid Search")
            with tprint_timer("Coarse Grid Search", level="PERFORMANCE"):
                coarse_results = self._coarse_grid_search(
                    data_array,
                    search_space,
                    n_trials=self.tuning_config.coarse_grid_trials
                )
            
            # Stage 2: Fine Grid Search
            if self.best_params is not None:
                tprint_info("🔍 Stage 2: Fine Grid Search")
                with tprint_timer("Fine Grid Search", level="PERFORMANCE"):
                    fine_results = self._fine_grid_search(
                        data_array,
                        search_space,
                        best_params=self.best_params,
                        n_trials=self.tuning_config.fine_grid_trials
                    )
            
            # Stage 3: TPE Optimization
            tprint_info("🎯 Stage 3: TPE Optimization")
            with tprint_timer("TPE Optimization", level="PERFORMANCE"):
                tpe_results = self._tpe_optimization(
                    data_array,
                    search_space,
                    n_trials=self.tuning_config.tpe_trials
                )
        else:
            # Direct TPE optimization
            tprint_info("🎯 Direct TPE Optimization")
            with tprint_timer("TPE Optimization", level="PERFORMANCE"):
                tpe_results = self._tpe_optimization(
                    data_array,
                    search_space,
                    n_trials=self.tuning_config.n_trials
                )
        
        # Generate summary
        summary = self._generate_summary()
        
        tprint_success(f"🎉 Auto-Tuning Complete!")
        tprint_structured(summary, level="INFO")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_result': self.best_result,
            'trial_history': self.trial_history,
            'optimization_summary': summary
        }
    
    def auto_tune_hierarchical(
        self,
        data: pd.DataFrame,
        n_trials_per_group: Optional[int] = None,
        timeout_minutes: Optional[float] = None,
        use_adaptive_bounds: bool = True
    ) -> Dict[str, Any]:
        """
        Automatically tune MS-DR clustering using hierarchical optimization.
        
        This method uses hierarchical parameter optimization to reduce search
        space and improve convergence speed.
        
        Benefits over standard auto_tune:
        - 50-70% faster optimization
        - Better parameter exploration
        - Optimizes high-impact parameters first
        - More interpretable results
        
        Args:
            data: Market data DataFrame
            n_trials_per_group: Trials per parameter group (uses config default if None)
            timeout_minutes: Timeout in minutes (uses config default if None)
            use_adaptive_bounds: Adapt parameter bounds based on data characteristics
            
        Returns:
            Dictionary with tuning results:
                - best_params: Best hyperparameters found
                - best_score: Best composite quality score achieved
                - hierarchical_results: Full hierarchical optimization results
                - trial_history: History of all trials
                - optimization_summary: Summary statistics
        """
        if not HIERARCHICAL_HPO_AVAILABLE:
            tprint_warning("⚠️ Hierarchical HPO not available, falling back to standard auto_tune")
            return self.auto_tune(data, timeout_minutes=timeout_minutes)
        
        tprint_info("🚀 Starting Hierarchical MS-DR Auto-Tuning")
        
        # Override config if specified
        if n_trials_per_group is not None:
            self.tuning_config.n_trials_per_group = n_trials_per_group
        if timeout_minutes is not None:
            self.tuning_config.timeout_minutes = timeout_minutes
        
        # Log configuration
        tprint_structured({
            'n_trials_per_group': self.tuning_config.n_trials_per_group,
            'timeout_minutes': self.tuning_config.timeout_minutes,
            'use_adaptive_bounds': use_adaptive_bounds,
            'data_shape': data.shape
        }, level="INFO")
        
        # Convert data to numpy if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Reset history
        self.trial_history = []
        self.best_score = float('-inf')
        self.best_params = None
        self.best_result = None
        
        # Create parameter groups (adaptive or default)
        if use_adaptive_bounds:
            tprint_info("📊 Using adaptive parameter bounds based on data")
            hierarchical_opt = MSDRHierarchicalOptimizer(
                objective_func=lambda params: self._evaluate_params(params, data_array)
            )
            param_groups = hierarchical_opt.get_adaptive_search_space(data_array)
        else:
            param_groups = create_msdr_parameter_groups()
        
        # Create optimization stages
        stages = create_msdr_optimization_stages(
            n_trials_per_group=self.tuning_config.n_trials_per_group
        )
        
        # Create hierarchical optimizer
        hierarchical_optimizer = MSDRHierarchicalOptimizer(
            objective_func=lambda params: self._evaluate_params(params, data_array),
            param_groups=param_groups,
            stages=stages
        )
        
        # Run hierarchical optimization
        with tprint_timer("Hierarchical Optimization", level="PERFORMANCE"):
            hierarchical_results = hierarchical_optimizer.optimize(
                data=data_array,
                timeout_minutes=self.tuning_config.timeout_minutes,
                n_trials_per_group=self.tuning_config.n_trials_per_group,
                show_progress=True
            )
        
        # Update best results from hierarchical optimization
        self.best_params = hierarchical_results.get('best_params', {})
        self.best_score = hierarchical_results.get('best_score', float('-inf'))
        
        # Generate summary
        summary = self._generate_summary()
        summary['optimization_method'] = 'hierarchical'
        summary['groups_optimized'] = len(param_groups)
        
        tprint_success(f"🎉 Hierarchical Auto-Tuning Complete!")
        tprint_structured(summary, level="INFO")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_result': self.best_result,
            'trial_history': self.trial_history,
            'hierarchical_results': hierarchical_results,
            'optimization_summary': summary
        }
    
    def _coarse_grid_search(
        self,
        data: np.ndarray,
        search_space: Dict[str, Dict[str, Any]],
        n_trials: int
    ) -> Dict[str, Any]:
        """Stage 1: Coarse grid search."""
        tprint_info(f"  Evaluating {n_trials} coarse grid points...")
        
        # Build coarse grid
        coarse_grid = build_coarse_grid_from_search_space(
            search_space,
            grid_points=self.tuning_config.coarse_grid_points
        )
        
        # Limit grid size
        if len(coarse_grid) > n_trials:
            np.random.seed(self.tuning_config.random_state)
            indices = np.random.choice(len(coarse_grid), n_trials, replace=False)
            coarse_grid = [coarse_grid[i] for i in indices]
        
        # Evaluate each point
        scores = []
        for i, params in enumerate(coarse_grid):
            tprint_debug(f"    Coarse trial {i+1}/{len(coarse_grid)}")
            score = self._evaluate_params(params, data)
            scores.append(score)
        
        # Report results with proper handling of empty/invalid scores
        if scores:
            valid_scores = [s for s in scores if s != float('-inf')]
            if valid_scores:
                tprint_success(f"  ✅ Coarse grid completed: Best score = {max(valid_scores):.4f}")
            else:
                tprint_warning(f"  ⚠️ Coarse grid completed: No valid scores obtained")
        else:
            tprint_error(f"  ❌ Coarse grid failed: No trials completed")
        
        return {
            'grid': coarse_grid,
            'scores': scores,
            'best_score': max(scores)
        }
    
    def _fine_grid_search(
        self,
        data: np.ndarray,
        search_space: Dict[str, Dict[str, Any]],
        best_params: Dict[str, Any],
        n_trials: int
    ) -> Dict[str, Any]:
        """Stage 2: Fine grid search around best parameters."""
        tprint_info(f"  Refining around best params: {best_params}")
        
        # Build fine grid around best params
        fine_grid = build_fine_grid_around_best(
            search_space,
            best_params,
            grid_points=self.tuning_config.fine_grid_points
        )
        
        # Limit grid size
        if len(fine_grid) > n_trials:
            np.random.seed(self.tuning_config.random_state)
            indices = np.random.choice(len(fine_grid), n_trials, replace=False)
            fine_grid = [fine_grid[i] for i in indices]
        
        # Evaluate each point
        scores = []
        for i, params in enumerate(fine_grid):
            tprint_debug(f"    Fine trial {i+1}/{len(fine_grid)}")
            score = self._evaluate_params(params, data)
            scores.append(score)
        
        # Report results with proper handling of empty/invalid scores
        if scores:
            valid_scores = [s for s in scores if s != float('-inf')]
            if valid_scores:
                tprint_success(f"  ✅ Fine grid completed: Best score = {max(valid_scores):.4f}")
            else:
                tprint_warning(f"  ⚠️ Fine grid completed: No valid scores obtained")
        else:
            tprint_error(f"  ❌ Fine grid failed: No trials completed")
        
        return {
            'grid': fine_grid,
            'scores': scores,
            'best_score': max(scores)
        }
    
    def _tpe_optimization(
        self,
        data: np.ndarray,
        search_space: Dict[str, Dict[str, Any]],
        n_trials: int
    ) -> Dict[str, Any]:
        """Stage 3: TPE-based Bayesian optimization."""
        try:
            import optuna
            from optuna.samplers import TPESampler
            
            tprint_info(f"  Running {n_trials} TPE trials...")
            
            # Create objective function for Optuna
            def objective(trial):
                params = {}
                
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Evaluate parameters
                score = self._evaluate_params(params, data)
                return score
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.tuning_config.random_state)
            )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=self.tuning_config.timeout_minutes * 60 if self.tuning_config.timeout_minutes else None,
                show_progress_bar=True
            )
            
            tprint_success(f"  ✅ TPE optimization completed: Best score = {study.best_value:.4f}")
            
            return {
                'study': study,
                'best_params': study.best_params,
                'best_score': study.best_value
            }
            
        except ImportError:
            tprint_warning("⚠️ Optuna not available, skipping TPE optimization")
            return {}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate optimization summary."""
        if not self.trial_history:
            return {}
        
        scores = [trial['composite_score'] for trial in self.trial_history]
        
        summary = {
            'total_trials': len(self.trial_history),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'score_range': (min(scores), max(scores)),
            'improvement': (self.best_score - scores[0]) if (len(scores) > 0 and scores[0] != float('-inf')) else 0.0
        }
        
        return summary


# Convenience function
def auto_tune_ms_dr_clustering(
    data: pd.DataFrame,
    n_trials: int = 100,
    timeout_minutes: float = 60.0,
    enable_staged_optimization: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for MS-DR auto-tuning.
    
    Args:
        data: Market data DataFrame
        n_trials: Total number of trials
        timeout_minutes: Timeout in minutes
        enable_staged_optimization: Use coarse -> fine -> TPE strategy
        
    Returns:
        Tuning results dictionary
        
    Example:
        >>> result = auto_tune_ms_dr_clustering(
        ...     data=market_data,
        ...     n_trials=100,
        ...     timeout_minutes=60.0
        ... )
        >>> best_params = result['best_params']
        >>> best_score = result['best_score']
    """
    tuner = MSDRAutoTuner()
    return tuner.auto_tune(
        data=data,
        n_trials=n_trials,
        timeout_minutes=timeout_minutes,
        enable_staged_optimization=enable_staged_optimization
    )


__all__ = [
    'MSDRAutoTuner',
    'MSDRTuningConfig',
    'auto_tune_ms_dr_clustering'
]
