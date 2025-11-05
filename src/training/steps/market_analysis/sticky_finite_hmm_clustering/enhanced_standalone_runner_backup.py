"""
Enhanced Standalone Sticky Finite HMM Clustering Runner with Auto-Tuning

This module provides an enhanced standalone runner with:
- 2-stage optimization (grid search -> fine grid search)
- Multi-objective optimization support
- Quality assessor integration
- Composite scoring and KPI tracking
- Advanced SVI optimizations (natural gradients, Rao-Blackwellization)

Usage Example:
    ```python
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering.enhanced_standalone_runner import (
        run_sticky_finite_hmm_with_auto_tuning
    )
    
    # Basic auto-tuning
    results = run_sticky_finite_hmm_with_auto_tuning(
        market_data=df,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        optimization_stages=2,  # grid -> fine grid
        use_multi_objective=False
    )
    
    # Multi-objective optimization
    results = run_sticky_finite_hmm_with_auto_tuning(
        market_data=df,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        optimization_stages=2,
        use_multi_objective=True,
        objectives=["composite_score", "silhouette_score", "transition_persistence"]
    )
    ```
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import core components
try:
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
        StickyFiniteHMMClusterer,
        StickyFiniteHMMConfig
    )
    CLUSTERER_AVAILABLE = True
except ImportError:
    CLUSTERER_AVAILABLE = False
    StickyFiniteHMMClusterer = None
    StickyFiniteHMMConfig = None

# Import quality assessment
try:
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
        ClusterQualityAssessor,
        create_cluster_quality_assessor
    )
    _quality_assessor_available = True
except ImportError:
    _quality_assessor_available = False
    ClusterQualityAssessor = None
    def create_cluster_quality_assessor(**kwargs):
        return None

# Make available for the rest of the code
QUALITY_ASSESSOR_AVAILABLE = _quality_assessor_available

try:
    from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
        DEFAULT_CLUSTERING_GOALS,
        DEFAULT_OPTIMIZATION_TARGETS
    )
except ImportError:
    pass

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space,
        build_fine_grid_around_best
    )
    _grid_utils_available = True
except ImportError:
    _grid_utils_available = False
    def build_coarse_grid_from_search_space(*args, **kwargs):
        return []
    def build_fine_grid_around_best(*args, **kwargs):
        return []

# Make available for the rest of the code
GRID_UTILS_AVAILABLE = _grid_utils_available

# Import multi-objective optimization
try:
    from src.utils.ml_common.optimization.pareto import (
        ParetoOptimizer,
        Solution,
        compute_pareto_front,
        select_knee_point,
        ObjectiveDirection
    )
    _pareto_available = True
except ImportError:
    _pareto_available = False
    ParetoOptimizer = None
    Solution = None
    compute_pareto_front = None
    select_knee_point = None
    ObjectiveDirection = None

# Make available for the rest of the code
PARETO_AVAILABLE = _pareto_available

# Import utilities
try:
    from src.utils.tprint import tprint_error, tprint_info, tprint_success, tprint_warning, tprint_structured, tprint_timer
except ImportError:
    # Fallback print functions
    def tprint_error(msg):
        print(f"❌ ERROR: {msg}")
    def tprint_info(msg):
        print(f"ℹ️  INFO: {msg}")
    def tprint_success(msg):
        print(f"✅ SUCCESS: {msg}")
    def tprint_warning(msg):
        print(f"⚠️  WARNING: {msg}")
    def tprint_structured(data, level="INFO"):
        print(f"📊 {data}")
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    def tprint_structured(data, level="INFO"): print(f"📊 {data}")
    def tprint_timer(name, level="INFO"): 
        import contextlib
        return contextlib.nullcontext()


@dataclass
class AutoTuningConfig:
    """Configuration for auto-tuning."""
    optimization_stages: int = 2  # grid -> fine grid
    use_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["composite_score"])
    max_trials_per_stage: int = 50
    timeout_seconds: int = 1800
    grid_resolution: Dict[str, int] = field(default_factory=lambda: {
        'K': 3, 'base_alpha': 3, 'kappa': 3, 'num_iters': 2, 'lr': 3
    })
    fine_grid_factor: float = 0.5  # Fine grid range around best params
    enable_kpi_tracking: bool = True
    save_all_trials: bool = True


@dataclass
class OptimizationResult:
    """Result of optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_objectives: Dict[str, float]
    all_trials: List[Dict[str, Any]]
    pareto_solutions: Optional[List[Solution]] = None
    optimization_time: float = 0.0
    kpi_metrics: Dict[str, Any] = field(default_factory=dict)
    stage_results: List[Dict[str, Any]] = field(default_factory=list)
    final_clustering_results: Optional[Dict[str, Any]] = None


class EnhancedStandaloneRunner:
    """Enhanced standalone runner with auto-tuning capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced runner."""
        self.config = config or {}
        self.quality_assessor = None
        self.kpi_tracker = {}
        
        # Initialize quality assessor if available
        if QUALITY_ASSESSOR_AVAILABLE:
            self.quality_assessor = create_cluster_quality_assessor(
                enable_hardware_optimization=True,
                enable_vectorization=True
            )
    
    def _get_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter search space for optimization."""
        return {
            'K': {
                'type': 'categorical',
                'values': [3, 4, 5, 6, 7, 8]
            },
            'base_alpha': {
                'type': 'float',
                'low': 0.1,
                'high': 2.0,
                'log': True
            },
            'kappa': {
                'type': 'float',
                'low': 1.0,
                'high': 50.0,
                'log': True
            },
            'num_iters': {
                'type': 'categorical',
                'values': [400, 600, 800, 1000, 1200]
            },
            'lr': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-1,
                'log': True
            },
            'n_mixtures': {
                'type': 'categorical',
                'values': [1, 2, 3]
            },
            'pca_components': {
                'type': 'categorical',
                'values': [10, 15, 20, 25]
            }
        }
    
    def _evaluate_objectives(self, results: Dict[str, Any], objectives: List[str]) -> Dict[str, float]:
        """Evaluate multiple optimization objectives."""
        objectives_scores = {}
        
        quality_metrics = results.get('quality_metrics', {})
        
        for obj in objectives:
            if obj == 'composite_score':
                objectives_scores[obj] = quality_metrics.get('composite_score', 0.0)
            elif obj == 'silhouette_score':
                objectives_scores[obj] = quality_metrics.get('silhouette_score', 0.0)
            elif obj == 'davies_bouldin_score':
                # Lower is better, so negate for maximization
                objectives_scores[obj] = -quality_metrics.get('davies_bouldin_score', 10.0)
            elif obj == 'calinski_harabasz_score':
                objectives_scores[obj] = quality_metrics.get('calinski_harabasz_score', 0.0)
            elif obj == 'transition_persistence':
                objectives_scores[obj] = quality_metrics.get('transition_persistence', 0.0)
            elif obj == 'final_elbo':
                objectives_scores[obj] = results.get('final_elbo', 0.0)
            else:
                objectives_scores[obj] = 0.0
        
        return objectives_scores
    
    def _run_grid_search(
        self, 
        market_data: pd.DataFrame,
        search_space: Dict[str, Dict[str, Any]],
        config: AutoTuningConfig,
        stage_name: str = "Grid Search"
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Run grid search optimization."""
        tprint_info(f"🔍 Starting {stage_name}")
        
        if not GRID_UTILS_AVAILABLE:
            # Fallback to simple grid sampling
            trials = []
            best_trial = None
            best_score = -np.inf
            
            # Generate simple grid samples
            for i in range(min(config.max_trials_per_stage, 20)):
                params = {
                    'K': np.random.choice(search_space['K']['values']),
                    'base_alpha': np.random.uniform(0.1, 2.0),
                    'kappa': np.random.uniform(1.0, 50.0),
                    'num_iters': np.random.choice(search_space['num_iters']['values']),
                    'lr': np.random.uniform(1e-4, 1e-1),
                    'n_mixtures': np.random.choice(search_space['n_mixtures']['values']),
                    'pca_components': np.random.choice(search_space['pca_components']['values'])
                }
                
                # Evaluate parameters
                trial_result = self._evaluate_parameters(market_data, params, config.objectives)
                trials.append(trial_result)
                
                # Update best
                if trial_result['score'] > best_score:
                    best_score = trial_result['score']
                    best_trial = trial_result
            
            return trials, best_trial
        
        # Use grid utilities if available
        if stage_name == "Grid Search":
            param_grid = build_coarse_grid_from_search_space(
                search_space, 
                resolution=config.grid_resolution
            )
        else:  # Fine grid search
            # Need best_params from previous stage
            best_params = getattr(config, 'previous_best_params', {})
            param_grid = build_fine_grid_around_best(
                search_space,
                best_params,
                grid_points=10  # Number of fine grid points per parameter
            )
        
        trials = []
        best_trial = None
        best_score = -np.inf
        
        for i, params in enumerate(param_grid[:config.max_trials_per_stage]):
            tprint_info(f"  Trial {i+1}/{min(len(param_grid), config.max_trials_per_stage)}: K={params['K']}")
            
            # Evaluate parameters
            trial_result = self._evaluate_parameters(market_data, params, config.objectives)
            trials.append(trial_result)
            
            # Update best
            if trial_result['score'] > best_score:
                best_score = trial_result['score']
                best_trial = trial_result
        
        return trials, best_trial
    
    def _evaluate_parameters(
        self, 
        market_data: pd.DataFrame,
        params: Dict[str, Any],
        objectives: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a set of parameters."""
        try:
            # Run clustering with parameters
            results = run_sticky_finite_hmm_clustering(
                market_data=market_data,
                K=params['K'],
                base_alpha=params['base_alpha'],
                kappa=params['kappa'],
                num_iters=params['num_iters'],
                lr=params['lr'],
                n_mixtures=params['n_mixtures'],
                pca_components=params['pca_components'],
                save_results=False,  # Don't save during optimization
                compute_posteriors=True
            )
            
            # Evaluate objectives
            objectives_scores = self._evaluate_objectives(results, objectives)
            
            # Composite score (weighted average)
            if len(objectives) == 1:
                score = objectives_scores[objectives[0]]
            else:
                # Simple average for multi-objective (will be handled by Pareto)
                score = np.mean(list(objectives_scores.values()))
            
            return {
                'params': params.copy(),
                'score': score,
                'objectives': objectives_scores,
                'results': results,
                'success': True
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Trial failed: {e}")
            return {
                'params': params.copy(),
                'score': -np.inf,
                'objectives': {obj: -np.inf for obj in objectives},
                'results': None,
                'success': False,
                'error': str(e)
            }
    
    def _run_multi_objective_optimization(
        self,
        trials: List[Dict[str, Any]],
        objectives: List[str]
    ) -> Tuple[List[Solution], Solution]:
        """Run multi-objective optimization using Pareto front."""
        if not PARETO_AVAILABLE:
            tprint_warning("⚠️ Pareto optimization not available, using single objective")
            return [], None
        
        # Convert trials to solutions
        solutions = []
        for trial in trials:
            if trial['success']:
                solution = Solution(
                    params=trial['params'],
                    objectives=trial['objectives'],
                    score=trial['score']
                )
                solutions.append(solution)
        
        # Define objective directions (all maximize for now)
        objective_directions = {obj: ObjectiveDirection.MAXIMIZE for obj in objectives}
        
        # Compute Pareto front
        pareto_front = compute_pareto_front(solutions, objective_directions)
        
        # Select knee point as best solution
        best_solution = select_knee_point(pareto_front) if pareto_front else None
        
        return pareto_front, best_solution
    
    def _update_kpi_tracker(self, stage_results: Dict[str, Any]):
        """Update KPI tracking metrics."""
        if not self.kpi_tracker:
            self.kpi_tracker = {
                'total_trials': 0,
                'successful_trials': 0,
                'failed_trials': 0,
                'best_score_history': [],
                'stage_times': [],
                'objective_improvements': {}
            }
        
        self.kpi_tracker['total_trials'] += stage_results.get('trials_evaluated', 0)
        self.kpi_tracker['successful_trials'] += stage_results.get('successful_trials', 0)
        self.kpi_tracker['failed_trials'] += stage_results.get('failed_trials', 0)
        self.kpi_tracker['best_score_history'].append(stage_results.get('best_score', 0.0))
        self.kpi_tracker['stage_times'].append(stage_results.get('stage_time', 0.0))
    
    def run_auto_tuning(
        self,
        market_data: pd.DataFrame,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "1h",
        auto_tuning_config: Optional[AutoTuningConfig] = None
    ) -> OptimizationResult:
        """
        Run auto-tuning with 2-stage optimization.
        
        Args:
            market_data: Market data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            auto_tuning_config: Auto-tuning configuration
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        config = auto_tuning_config or AutoTuningConfig()
        
        tprint_info("🚀 Starting Enhanced Auto-Tuning with 2-Stage Optimization")
        tprint_structured({
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "optimization_stages": config.optimization_stages,
            "use_multi_objective": config.use_multi_objective,
            "objectives": config.objectives,
            "max_trials_per_stage": config.max_trials_per_stage
        })
        
        start_time = time.time()
        search_space = self._get_search_space()
        all_trials = []
        stage_results = []
        best_params = {}
        best_score = -np.inf
        best_objectives = {}
        pareto_solutions = None
        
        with tprint_timer("Auto-Tuning"):
            # Stage 1: Coarse Grid Search
            if config.optimization_stages >= 1:
                stage_start = time.time()
                trials, best_trial = self._run_grid_search(
                    market_data, search_space, config, "Coarse Grid Search"
                )
                stage_time = time.time() - stage_start
                
                all_trials.extend(trials)
                
                if best_trial:
                    best_params = best_trial['params']
                    best_score = best_trial['score']
                    best_objectives = best_trial['objectives']
                
                stage_result = {
                    'stage': 1,
                    'stage_name': 'Coarse Grid Search',
                    'trials_evaluated': len(trials),
                    'successful_trials': sum(1 for t in trials if t['success']),
                    'failed_trials': sum(1 for t in trials if not t['success']),
                    'best_score': best_score,
                    'best_params': best_params.copy(),
                    'stage_time': stage_time
                }
                stage_results.append(stage_result)
                self._update_kpi_tracker(stage_result)
                
                tprint_success(f"✅ Stage 1 Complete: Best Score = {best_score:.4f}")
            
            # Stage 2: Fine Grid Search
            if config.optimization_stages >= 2 and best_params:
                stage_start = time.time()
                
                # Set previous best for fine grid generation
                # Store in config object for fine grid generation
                if not hasattr(config, 'previous_best_params'):
                    config.previous_best_params = best_params
                
                trials, best_trial = self._run_grid_search(
                    market_data, search_space, config, "Fine Grid Search"
                )
                stage_time = time.time() - stage_start
                
                all_trials.extend(trials)
                
                if best_trial and best_trial['score'] > best_score:
                    best_params = best_trial['params']
                    best_score = best_trial['score']
                    best_objectives = best_trial['objectives']
                
                stage_result = {
                    'stage': 2,
                    'stage_name': 'Fine Grid Search',
                    'trials_evaluated': len(trials),
                    'successful_trials': sum(1 for t in trials if t['success']),
                    'failed_trials': sum(1 for t in trials if not t['success']),
                    'best_score': best_score,
                    'best_params': best_params.copy(),
                    'stage_time': stage_time
                }
                stage_results.append(stage_result)
                self._update_kpi_tracker(stage_result)
                
                tprint_success(f"✅ Stage 2 Complete: Best Score = {best_score:.4f}")
            
            # Multi-objective optimization if enabled
            if config.use_multi_objective and len(config.objectives) > 1:
                tprint_info("🎯 Running Multi-Objective Optimization")
                pareto_solutions, best_solution = self._run_multi_objective_optimization(
                    all_trials, config.objectives
                )
                
                if best_solution:
                    best_params = best_solution.params
                    best_objectives = best_solution.objectives
                    tprint_success(f"✅ Multi-Objective Best Solution Found")
        
        optimization_time = time.time() - start_time
        
        # KPI metrics
        kpi_metrics = {}
        if config.enable_kpi_tracking:
            kpi_metrics = {
                'total_optimization_time': optimization_time,
                'total_trials': len(all_trials),
                'success_rate': self.kpi_tracker.get('successful_trials', 0) / max(len(all_trials), 1),
                'best_score_improvement': (
                    self.kpi_tracker['best_score_history'][-1] - self.kpi_tracker['best_score_history'][0]
                    if len(self.kpi_tracker['best_score_history']) > 1 else 0.0
                ),
                'average_stage_time': np.mean(self.kpi_tracker.get('stage_times', [0])),
                'trials_per_second': len(all_trials) / max(optimization_time, 1)
            }
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_objectives=best_objectives,
            all_trials=all_trials,
            pareto_solutions=pareto_solutions,
            optimization_time=optimization_time,
            kpi_metrics=kpi_metrics,
            stage_results=stage_results
        )
        
        # Print summary
        tprint_info("=" * 60)
        tprint_info("ENHANCED AUTO-TUNING RESULTS")
        tprint_info("=" * 60)
        tprint_structured({
            "best_score": best_score,
            "best_params": best_params,
            "objectives": best_objectives,
            "total_trials": len(all_trials),
            "optimization_time": optimization_time,
            "success_rate": kpi_metrics.get('success_rate', 0.0)
        })
        tprint_info("=" * 60)
        
        return result


# Global runner instance
_enhanced_runner = None

def get_enhanced_runner() -> EnhancedStandaloneRunner:
    """Get or create enhanced runner instance."""
    global _enhanced_runner
    if _enhanced_runner is None:
        _enhanced_runner = EnhancedStandaloneRunner()
    return _enhanced_runner


@dataclass
class AutoTuningConfig:
    """Configuration for enhanced auto-tuning."""
    optimization_stages: int = 2
    use_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["composite_score"])
    max_trials_per_stage: int = 50
    timeout_seconds: int = 1800
    enable_kpi_tracking: bool = True
    save_results: bool = True
    grid_resolution: int = 3
    fine_grid_factor: float = 0.2
    
    # Runtime state (not part of config)
    previous_best_params: Dict[str, Any] = field(default_factory=dict)


def run_sticky_finite_hmm_with_auto_tuning(
    market_data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    optimization_stages: int = 2,
    use_multi_objective: bool = False,
    objectives: Optional[List[str]] = None,
    max_trials_per_stage: int = 50,
    timeout_seconds: int = 1800,
    enable_kpi_tracking: bool = True,
    save_results: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Convenience function for auto-tuning with enhanced features.
    
    Args:
        market_data: Market data DataFrame
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        optimization_stages: Number of optimization stages (1=grid, 2=grid+fine)
        use_multi_objective: Enable multi-objective optimization
        objectives: List of objectives to optimize
        max_trials_per_stage: Maximum trials per optimization stage
        timeout_seconds: Optimization timeout
        enable_kpi_tracking: Enable KPI tracking
        save_results: Save final results to artifacts
        **kwargs: Additional parameters
        
    Returns:
        OptimizationResult with best parameters and metrics
    """
    if objectives is None:
        objectives = ["composite_score"]
    
    config = AutoTuningConfig(
        optimization_stages=optimization_stages,
        use_multi_objective=use_multi_objective,
        objectives=objectives,
        max_trials_per_stage=max_trials_per_stage,
        timeout_seconds=timeout_seconds,
        enable_kpi_tracking=enable_kpi_tracking,
        save_all_trials=True
    )
    
    runner = get_enhanced_runner()
    result = runner.run_auto_tuning(
        market_data=market_data,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        auto_tuning_config=config
    )
    
    # Save final results if requested
    if save_results and result.best_params:
        tprint_info("💾 Running final clustering with best parameters...")
        final_results = run_sticky_finite_hmm_clustering(
            market_data=market_data,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            save_results=True,
            **result.best_params,
            **kwargs
        )
        
        # Add final results to optimization result
        result.final_clustering_results = final_results
    
    return result


__all__ = [
    'EnhancedStandaloneRunner',
    'AutoTuningConfig',
    'OptimizationResult',
    'run_sticky_finite_hmm_with_auto_tuning',
    'get_enhanced_runner'
]
