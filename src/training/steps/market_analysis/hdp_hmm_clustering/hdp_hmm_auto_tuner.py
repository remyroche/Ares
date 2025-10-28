"""
HDP-HMM Clustering Auto-Tuner

This module provides automatic hyperparameter tuning for HDP-HMM clustering using
a multi-stage optimization approach:
1. Coarse Grid Search - Broad exploration of parameter space
2. Fine Grid Search - Refinement around best coarse results
3. TPE (Bayesian Optimization) - Final optimization with Optuna

The objective is to maximize the composite_score from cluster_quality_assessor.py
which combines multiple quality metrics (silhouette, DBI, CV ratio, balance, temporal).

Usage:
    ```python
    from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_auto_tuning
    
    best_params, best_score, tuning_results = run_hdp_hmm_auto_tuning(
        market_data=df,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        n_trials=100,
        timeout=3600
    )
    ```
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import logging

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured, tprint_timer, tprint_performance
)

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        OptimizationConfig,
        OPTUNA_AVAILABLE
    )
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space,
        build_fine_grid_around_best
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Optimization utilities not fully available: {e}")
    OPTIMIZATION_AVAILABLE = False
    OPTUNA_AVAILABLE = False

# Import HDP-HMM components
from .hdp_hmm_clusterer import HDPHMMClusterer, HDPHMMConfig
from .standalone_runner import run_hdp_hmm_clustering

# Import quality assessment
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    create_cluster_quality_assessor
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS
)

# Import artifact manager
try:
    from src.utils.artifact_manager import ArtifactManager
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError:
    ARTIFACT_MANAGER_AVAILABLE = False
    tprint_warning("⚠️ Artifact manager not available")


@dataclass
class HDPHMMSearchSpace:
    """
    Search space definition for HDP-HMM hyperparameters.
    
    Parameters:
        alpha: HDP concentration parameter
            - Controls regime diversity (higher = more regimes)
            - Range: 1.0 - 10.0
            - Default range: 2.0 - 5.0
            
        kappa: Stickiness parameter
            - Controls regime persistence (higher = longer durations)
            - Range: 10.0 - 100.0
            - Default range: 30.0 - 70.0
            
        gamma: Base distribution hyperparameter
            - Controls prior over states
            - Range: 1.0 - 10.0
            - Default range: 2.0 - 5.0
            
        n_iterations: Number of Gibbs sampling iterations
            - More iterations = better convergence but slower
            - Range: 50 - 500
            - Default range: 100 - 200
            
        pca_components: Number of PCA components for dimensionality reduction
            - Reduces feature space complexity
            - Range: 5 - 20
            - Default range: 8 - 15
            
        min_features: Minimum features to select from feature bank
            - Ensures sufficient signal
            - Range: 20 - 100
            - Default range: 40 - 60
            
        max_features: Maximum features to select from feature bank
            - Prevents overfitting and reduces computation
            - Range: 50 - 150
            - Default range: 80 - 120
    """
    
    # Primary HDP-HMM parameters
    alpha_min: float = 2.0
    alpha_max: float = 5.0
    
    kappa_min: float = 30.0
    kappa_max: float = 70.0
    
    gamma_min: float = 2.0
    gamma_max: float = 5.0
    
    n_iterations_min: int = 100
    n_iterations_max: int = 200
    
    # Feature selection parameters
    min_features_min: int = 40
    min_features_max: int = 60
    
    max_features_min: int = 80
    max_features_max: int = 120
    
    # PCA parameters
    pca_components_min: int = 8
    pca_components_max: int = 15
    
    def to_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Convert to optimization search space format."""
        return {
            'alpha': {
                'type': 'float',
                'low': self.alpha_min,
                'high': self.alpha_max,
                'log': False
            },
            'kappa': {
                'type': 'float',
                'low': self.kappa_min,
                'high': self.kappa_max,
                'log': False
            },
            'gamma': {
                'type': 'float',
                'low': self.gamma_min,
                'high': self.gamma_max,
                'log': False
            },
            'n_iterations': {
                'type': 'int',
                'low': self.n_iterations_min,
                'high': self.n_iterations_max
            },
            'min_features': {
                'type': 'int',
                'low': self.min_features_min,
                'high': self.max_features_max
            },
            'max_features': {
                'type': 'int',
                'low': self.max_features_min,
                'high': self.max_features_max
            },
            'pca_components': {
                'type': 'int',
                'low': self.pca_components_min,
                'high': self.pca_components_max
            }
        }


@dataclass
class TuningResult:
    """Result from hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_score: float
    coarse_grid_results: List[Dict[str, Any]] = field(default_factory=list)
    fine_grid_results: List[Dict[str, Any]] = field(default_factory=list)
    tpe_results: List[Dict[str, Any]] = field(default_factory=list)
    total_time: float = 0.0
    n_trials: int = 0
    convergence_info: Dict[str, Any] = field(default_factory=dict)


class HDPHMMAutoTuner:
    """
    Auto-tuner for HDP-HMM clustering using multi-stage optimization.
    
    This class implements a three-stage optimization strategy:
    1. Coarse Grid Search: Broad exploration with sparse grid
    2. Fine Grid Search: Refinement around best coarse results
    3. TPE Optimization: Bayesian optimization for final tuning
    
    The objective function maximizes the composite_score from cluster quality
    assessment, which combines:
    - Silhouette score (cluster cohesion)
    - Davies-Bouldin index (cluster separation)
    - CV ratio (between/within variance)
    - Balance score (cluster size distribution)
    - Temporal smoothness (regime stability)
    """
    
    def __init__(self,
                 market_data: pd.DataFrame,
                 symbol: str = "ETHUSDT",
                 exchange: str = "binance",
                 timeframe: str = "1h",
                 search_space: Optional[HDPHMMSearchSpace] = None,
                 artifact_manager: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize auto-tuner.
        
        Args:
            market_data: Market data DataFrame with OHLCV columns
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            search_space: Custom search space (uses defaults if None)
            artifact_manager: Optional artifact manager for saving results
            logger: Optional logger
        """
        self.market_data = market_data
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.search_space = search_space or HDPHMMSearchSpace()
        self.artifact_manager = artifact_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Quality assessor
        self.quality_assessor = create_cluster_quality_assessor(artifact_manager)
        
        # Tracking
        self.trial_history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf
        
        tprint_info("🎯 HDP-HMM Auto-Tuner Initialized")
        tprint_structured({
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_shape": market_data.shape,
            "search_space": "custom" if search_space else "default"
        }, level="INFO")
    
    def objective_function(self, params: Dict[str, Any]) -> float:
        """
        Objective function to maximize: composite_score from cluster quality.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Composite score (higher is better)
        """
        try:
            # Validate min_features <= max_features
            if params['min_features'] > params['max_features']:
                params['min_features'] = params['max_features'] - 10
            
            # Run clustering with these parameters
            results = run_hdp_hmm_clustering(
                market_data=self.market_data,
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                min_features=int(params['min_features']),
                max_features=int(params['max_features']),
                alpha=float(params['alpha']),
                kappa=float(params['kappa']),
                gamma=float(params['gamma']),
                n_iterations=int(params['n_iterations']),
                pca_components=int(params.get('pca_components', 10)),
                enable_pca=True,
                save_results=False  # Don't save intermediate results
            )
            
            # Extract composite score
            composite_score = results['quality_metrics'].get('composite_score', 0.0)
            
            # Record trial
            trial_info = {
                'params': params.copy(),
                'score': composite_score,
                'n_clusters': results['n_clusters'],
                'quality_metrics': results['quality_metrics']
            }
            self.trial_history.append(trial_info)
            
            # Update best
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_params = params.copy()
                tprint_success(f"✨ New best score: {composite_score:.4f}")
                tprint_structured({
                    "alpha": params['alpha'],
                    "kappa": params['kappa'],
                    "n_clusters": results['n_clusters'],
                    "composite_score": composite_score
                }, level="INFO")
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Objective function failed: {e}")
            tprint_error(f"❌ Trial failed: {e}")
            return -np.inf
    
    def coarse_grid_search(self, n_points: int = 3) -> List[Dict[str, Any]]:
        """
        Stage 1: Coarse grid search for broad exploration.
        
        Args:
            n_points: Number of points per parameter (default: 3)
            
        Returns:
            List of trial results
        """
        tprint_info("=" * 60)
        tprint_info("STAGE 1: COARSE GRID SEARCH")
        tprint_info("=" * 60)
        
        if not OPTIMIZATION_AVAILABLE:
            tprint_warning("⚠️ Optimization utilities not available, using defaults")
            return []
        
        # Build coarse grid
        search_space = self.search_space.to_search_space()
        grid_points = build_coarse_grid_from_search_space(search_space, n_points)
        
        tprint_info(f"🔍 Evaluating {len(grid_points)} parameter combinations")
        
        results = []
        with tprint_timer("Coarse Grid Search", level="PERFORMANCE"):
            for i, params in enumerate(grid_points, 1):
                tprint_info(f"Trial {i}/{len(grid_points)}")
                score = self.objective_function(params)
                results.append({
                    'params': params,
                    'score': score
                })
        
        tprint_success(f"✅ Coarse grid search completed: {len(results)} trials")
        return results
    
    def fine_grid_search(self, n_points: int = 3) -> List[Dict[str, Any]]:
        """
        Stage 2: Fine grid search around best coarse results.
        
        Args:
            n_points: Number of points per parameter (default: 3)
            
        Returns:
            List of trial results
        """
        tprint_info("=" * 60)
        tprint_info("STAGE 2: FINE GRID SEARCH")
        tprint_info("=" * 60)
        
        if not OPTIMIZATION_AVAILABLE or self.best_params is None:
            tprint_warning("⚠️ Skipping fine grid search")
            return []
        
        # Build fine grid around best parameters
        search_space = self.search_space.to_search_space()
        grid_points = build_fine_grid_around_best(
            search_space, 
            self.best_params, 
            n_points
        )
        
        tprint_info(f"🔍 Evaluating {len(grid_points)} parameter combinations around best")
        tprint_structured({
            "best_params": self.best_params,
            "best_score": self.best_score
        }, level="INFO")
        
        results = []
        with tprint_timer("Fine Grid Search", level="PERFORMANCE"):
            for i, params in enumerate(grid_points, 1):
                tprint_info(f"Trial {i}/{len(grid_points)}")
                score = self.objective_function(params)
                results.append({
                    'params': params,
                    'score': score
                })
        
        tprint_success(f"✅ Fine grid search completed: {len(results)} trials")
        return results
    
    def tpe_optimization(self, n_trials: int = 50, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Stage 3: TPE (Tree-structured Parzen Estimator) Bayesian optimization.
        
        Args:
            n_trials: Number of TPE trials (default: 50)
            timeout: Optional timeout in seconds
            
        Returns:
            List of trial results
        """
        tprint_info("=" * 60)
        tprint_info("STAGE 3: TPE BAYESIAN OPTIMIZATION")
        tprint_info("=" * 60)
        
        if not OPTUNA_AVAILABLE:
            tprint_warning("⚠️ Optuna not available, skipping TPE optimization")
            return []
        
        import optuna
        from optuna.samplers import TPESampler
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, multivariate=True)
        )
        
        # Define objective for Optuna
        def optuna_objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', self.search_space.alpha_min, self.search_space.alpha_max),
                'kappa': trial.suggest_float('kappa', self.search_space.kappa_min, self.search_space.kappa_max),
                'gamma': trial.suggest_float('gamma', self.search_space.gamma_min, self.search_space.gamma_max),
                'n_iterations': trial.suggest_int('n_iterations', self.search_space.n_iterations_min, self.search_space.n_iterations_max),
                'min_features': trial.suggest_int('min_features', self.search_space.min_features_min, self.search_space.min_features_max),
                'max_features': trial.suggest_int('max_features', self.search_space.max_features_min, self.search_space.max_features_max),
                'pca_components': trial.suggest_int('pca_components', self.search_space.pca_components_min, self.search_space.pca_components_max)
            }
            return self.objective_function(params)
        
        # Run optimization
        tprint_info(f"🔍 Running {n_trials} TPE trials")
        
        with tprint_timer("TPE Optimization", level="PERFORMANCE"):
            study.optimize(
                optuna_objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
        
        # Extract results
        results = []
        for trial in study.trials:
            results.append({
                'params': trial.params,
                'score': trial.value if trial.value is not None else -np.inf
            })
        
        tprint_success(f"✅ TPE optimization completed: {len(results)} trials")
        tprint_structured({
            "best_score": study.best_value,
            "best_params": study.best_params
        }, level="INFO")
        
        return results
    
    def run_full_tuning(self,
                       coarse_grid_points: int = 3,
                       fine_grid_points: int = 3,
                       tpe_trials: int = 50,
                       timeout: Optional[float] = None) -> TuningResult:
        """
        Run complete multi-stage tuning pipeline.
        
        Args:
            coarse_grid_points: Points per parameter in coarse grid (default: 3)
            fine_grid_points: Points per parameter in fine grid (default: 3)
            tpe_trials: Number of TPE trials (default: 50)
            timeout: Optional total timeout in seconds
            
        Returns:
            TuningResult with best parameters and scores
        """
        tprint_info("🚀 Starting Multi-Stage HDP-HMM Auto-Tuning")
        tprint_info("=" * 60)
        
        start_time = time.time()
        
        # Stage 1: Coarse Grid Search
        coarse_results = self.coarse_grid_search(n_points=coarse_grid_points)
        
        # Stage 2: Fine Grid Search
        fine_results = self.fine_grid_search(n_points=fine_grid_points)
        
        # Stage 3: TPE Optimization
        remaining_timeout = None
        if timeout is not None:
            elapsed = time.time() - start_time
            remaining_timeout = max(timeout - elapsed, 60)  # At least 1 minute
        
        tpe_results = self.tpe_optimization(n_trials=tpe_trials, timeout=remaining_timeout)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create result
        result = TuningResult(
            best_params=self.best_params,
            best_score=self.best_score,
            coarse_grid_results=coarse_results,
            fine_grid_results=fine_results,
            tpe_results=tpe_results,
            total_time=total_time,
            n_trials=len(self.trial_history),
            convergence_info={
                'n_coarse_trials': len(coarse_results),
                'n_fine_trials': len(fine_results),
                'n_tpe_trials': len(tpe_results)
            }
        )
        
        # Print summary
        tprint_info("=" * 60)
        tprint_info("TUNING COMPLETE")
        tprint_info("=" * 60)
        tprint_structured({
            "total_trials": result.n_trials,
            "total_time_seconds": total_time,
            "best_composite_score": self.best_score,
            "best_alpha": self.best_params['alpha'],
            "best_kappa": self.best_params['kappa'],
            "best_n_iterations": self.best_params['n_iterations'],
            "best_min_features": self.best_params['min_features'],
            "best_max_features": self.best_params['max_features']
        }, level="INFO")
        
        # Save results if artifact manager available
        if self.artifact_manager:
            try:
                self.artifact_manager.save(
                    data=result.best_params,
                    artifact_name="best_hdp_hmm_params",
                    artifact_type="metadata"
                )
                tprint_success("✅ Best parameters saved to artifacts")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save results: {e}")
        
        return result


def run_hdp_hmm_auto_tuning(
    market_data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    search_space: Optional[HDPHMMSearchSpace] = None,
    coarse_grid_points: int = 3,
    fine_grid_points: int = 3,
    tpe_trials: int = 50,
    timeout: Optional[float] = None,
    save_results: bool = True
) -> Tuple[Dict[str, Any], float, TuningResult]:
    """
    Run automatic hyperparameter tuning for HDP-HMM clustering.
    
    This function performs multi-stage optimization:
    1. Coarse grid search (3^7 = ~2187 combinations by default, pruned to coarse_grid_points^n_params)
    2. Fine grid search around best results
    3. TPE Bayesian optimization for final refinement
    
    Args:
        market_data: DataFrame with OHLCV columns
        symbol: Trading symbol (default: "ETHUSDT")
        exchange: Exchange name (default: "binance")
        timeframe: Timeframe (default: "1h" or "60m")
        search_space: Custom search space (uses defaults if None)
        coarse_grid_points: Points per parameter in coarse grid (default: 3)
        fine_grid_points: Points per parameter in fine grid (default: 3)
        tpe_trials: Number of TPE trials (default: 50)
        timeout: Optional total timeout in seconds
        save_results: Whether to save results to artifacts (default: True)
        
    Returns:
        Tuple of (best_params, best_score, tuning_result)
        
    Example:
        ```python
        import pandas as pd
        from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_auto_tuning
        
        # Load market data
        df = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)
        
        # Run auto-tuning
        best_params, best_score, results = run_hdp_hmm_auto_tuning(
            market_data=df,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1h",
            tpe_trials=100,
            timeout=3600  # 1 hour
        )
        
        print(f"Best composite score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Use best parameters for final clustering
        from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_clustering
        
        final_results = run_hdp_hmm_clustering(
            market_data=df,
            symbol="ETHUSDT",
            **best_params
        )
        ```
    """
    # Initialize artifact manager if saving results
    artifact_manager = None
    if save_results and ARTIFACT_MANAGER_AVAILABLE:
        config = {
            "paths": {
                "data_dir": "artifacts",
                "cache_dir": "data_cache",
                "reports_dir": "reports"
            }
        }
        artifact_manager = ArtifactManager(config)
        artifact_manager.set_context(
            step_name="hdp_hmm_auto_tuning",
            symbol=symbol,
            exchange=exchange,
            information="hyperparameter_optimization"
        )
    
    # Create tuner
    tuner = HDPHMMAutoTuner(
        market_data=market_data,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        search_space=search_space,
        artifact_manager=artifact_manager
    )
    
    # Run tuning
    tuning_result = tuner.run_full_tuning(
        coarse_grid_points=coarse_grid_points,
        fine_grid_points=fine_grid_points,
        tpe_trials=tpe_trials,
        timeout=timeout
    )
    
    return tuning_result.best_params, tuning_result.best_score, tuning_result


__all__ = [
    'HDPHMMSearchSpace',
    'HDPHMMAutoTuner',
    'TuningResult',
    'run_hdp_hmm_auto_tuning'
]
