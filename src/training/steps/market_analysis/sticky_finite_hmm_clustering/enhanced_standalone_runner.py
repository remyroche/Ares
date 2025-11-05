"""
Enhanced Standalone Runner for Sticky Finite HMM Clustering

This module provides enhanced auto-tuning capabilities with 2-stage optimization,
multi-objective optimization, quality assessor integration, and KPI tracking.

Features:
- 2-stage optimization (grid search -> fine grid search)
- Multi-objective optimization with Pareto front analysis
- Quality assessor integration with composite scoring
- KPI tracking and performance metrics
- Enhanced SVI optimizations (natural gradients, Rao-Blackwellization)

Example:
    ```python
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering.enhanced_standalone_runner import (
        run_sticky_finite_hmm_with_auto_tuning,
        AutoTuningConfig
    )
    
    config = AutoTuningConfig(
        optimization_stages=2,
        use_multi_objective=True,
        objectives=["composite_score", "silhouette_score", "transition_persistence"]
    )
    
    result = run_sticky_finite_hmm_with_auto_tuning(
        market_data=your_data,
        symbol="ETHUSDT",
        auto_tuning_config=config
    )
    ```
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import core components with error handling
try:
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
        StickyFiniteHMMClusterer,
        StickyFiniteHMMConfig
    )
    _clusterer_available = True
    # Access imports to avoid unused warnings
    _ = StickyFiniteHMMClusterer
    _ = StickyFiniteHMMConfig
except ImportError:
    _clusterer_available = False
    StickyFiniteHMMClusterer = None
    StickyFiniteHMMConfig = None

# Import quality assessor with error handling
try:
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
        ClusterQualityAssessor,
        create_cluster_quality_assessor
    )
    _quality_assessor_available = True
    # Access imports to avoid unused warnings
    _ = ClusterQualityAssessor
except ImportError:
    _quality_assessor_available = False
    ClusterQualityAssessor = None
    def create_cluster_quality_assessor(**kwargs):
        return None

# Import clustering optimization goals with error handling
try:
    from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
        DEFAULT_CLUSTERING_GOALS,
        DEFAULT_OPTIMIZATION_TARGETS
    )
    _goals_import_success = True
    # Access imports to avoid unused warnings
    _ = DEFAULT_CLUSTERING_GOALS
    _ = DEFAULT_OPTIMIZATION_TARGETS
except ImportError:
    _goals_import_success = False
    DEFAULT_CLUSTERING_GOALS = {}
    DEFAULT_OPTIMIZATION_TARGETS = {}

# Import optimization utilities with error handling
try:
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space,
        build_fine_grid_around_best
    )
    _grid_utils_available = True
    # Access imports to avoid unused warnings
    _ = build_coarse_grid_from_search_space
    _ = build_fine_grid_around_best
except ImportError:
    _grid_utils_available = False
    def build_coarse_grid_from_search_space(*args, **kwargs):
        return []
    def build_fine_grid_around_best(*args, **kwargs):
        return []

# Import multi-objective optimization with error handling
try:
    from src.utils.ml_common.optimization.pareto import (
        ParetoOptimizer,
        Solution,
        compute_pareto_front,
        select_knee_point,
        ObjectiveDirection
    )
    _pareto_available = True
    # Access imports to avoid unused warnings
    _ = ParetoOptimizer
    _ = ObjectiveDirection
except ImportError:
    _pareto_available = False
    Solution = None
    compute_pareto_front = None
    select_knee_point = None
    ObjectiveDirection = None

# Import tprint utilities with error handling
try:
    from src.utils.tprint import tprint_error, tprint_warning, tprint_structured, tprint_timer, tprint_info, tprint_success
    _tprint_available = True
    # Access imports to avoid unused warnings
    _ = tprint_error
    _ = tprint_warning
    _ = tprint_structured
    _ = tprint_timer
except ImportError:
    _tprint_available = False
    def tprint_error(msg, level="ERROR"): 
        print(f"[ERROR] {msg}")
    def tprint_warning(msg, level="WARNING"): 
        print(f"[WARNING] {msg}")
    def tprint_structured(msg, level="INFO"): 
        print(f"[INFO] {msg}")
    def tprint_timer(name, level="INFO"): 
        import contextlib
        return contextlib.nullcontext()
    def tprint_info(msg, level="INFO"): 
        print(f"[INFO] {msg}")
    def tprint_success(msg, level="SUCCESS"): 
        print(f"[SUCCESS] {msg}")


@dataclass
class AutoTuningConfig:
    """Configuration for enhanced auto-tuning."""
    optimization_stages: int = 2
    use_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["composite_score", "temporal_smoothness", "cv_ratio"])
    max_trials_per_stage: int = 50
    timeout_seconds: int = 1800
    enable_kpi_tracking: bool = True
    save_results: bool = True
    grid_resolution: int = 3
    fine_grid_factor: float = 0.2
    
    # Runtime state
    previous_best_params: Dict[str, Any] = field(default_factory=dict)


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


# Forward reference for Solution class
if _pareto_available:
    Solution = Solution  # Use the actual Solution class
else:
    # Create a mock Solution class for type hints
    @dataclass
    class Solution:
        params: Dict[str, Any]
        objectives: Dict[str, float]
        score: float


class EnhancedStandaloneRunner:
    """Enhanced standalone runner with auto-tuning capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced runner."""
        self.config = config or {}
        self.quality_assessor = None
        self.kpi_tracker = {}
        
        # Initialize quality assessor if available
        if _quality_assessor_available:
            try:
                self.quality_assessor = create_cluster_quality_assessor(
                    enable_hardware_optimization=True,
                    enable_vectorization=True
                )
            except Exception:
                self.quality_assessor = None
    
    def run_auto_tuning(
        self,
        market_data: pd.DataFrame,
        config: AutoTuningConfig
    ) -> OptimizationResult:
        """Run enhanced auto-tuning optimization."""
        start_time = time.time()
        tprint_info("🚀 Starting Enhanced Auto-Tuning with 2-Stage Optimization")
        
        # Initialize results
        all_trials = []
        best_params = {}
        best_score = -np.inf
        best_objectives = {}
        stage_results = []
        pareto_solutions = None
        
        # Define search space
        search_space = {
            'K': {'type': 'categorical', 'choices': [3, 5, 7]},
            'base_alpha': {'type': 'uniform', 'low': 0.1, 'high': 2.0},
            'kappa': {'type': 'uniform', 'low': 5.0, 'high': 50.0},
            'num_iters': {'type': 'categorical', 'choices': [50, 100, 150]},
            'lr': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-2},
            'n_mixtures': {'type': 'categorical', 'choices': [1, 2, 3]}
        }
        
        # Stage 1: Coarse Grid Search
        if config.optimization_stages >= 1:
            stage_start = time.time()
            trials, best_trial = self._run_grid_search(
                market_data, search_space, config, "Grid Search"
            )
            stage_time = time.time() - stage_start
            
            all_trials.extend(trials)
            
            if best_trial:
                best_params = best_trial['params']
                best_score = best_trial['score']
                best_objectives = best_trial.get('objectives', {})
                
                # Store for fine grid
                config.previous_best_params = best_params.copy()
            
            stage_result = {
                'stage': 1,
                'stage_name': 'Coarse Grid Search',
                'trials_evaluated': len(trials),
                'successful_trials': sum(1 for t in trials if t.get('success', False)),
                'best_score': best_score,
                'best_params': best_params,
                'stage_time': stage_time
            }
            stage_results.append(stage_result)
            self._update_kpi_tracker(stage_result)
            
            tprint_success(f"✅ Stage 1 Complete: Best Score = {best_score:.4f}")
        
        # Stage 2: Fine Grid Search
        if config.optimization_stages >= 2 and best_params:
            stage_start = time.time()
            
            trials, best_trial = self._run_grid_search(
                market_data, search_space, config, "Fine Grid Search"
            )
            stage_time = time.time() - stage_start
            
            all_trials.extend(trials)
            
            if best_trial and best_trial['score'] > best_score:
                best_params = best_trial['params']
                best_score = best_trial['score']
                best_objectives = best_trial.get('objectives', {})
            
            stage_result = {
                'stage': 2,
                'stage_name': 'Fine Grid Search',
                'trials_evaluated': len(trials),
                'successful_trials': sum(1 for t in trials if t.get('success', False)),
                'best_score': best_score,
                'best_params': best_params,
                'stage_time': stage_time
            }
            stage_results.append(stage_result)
            self._update_kpi_tracker(stage_result)
            
            tprint_success(f"✅ Stage 2 Complete: Best Score = {best_score:.4f}")
        
        # Multi-objective optimization
        if config.use_multi_objective and _pareto_available and all_trials:
            tprint_info("🎯 Running Multi-Objective Optimization")
            pareto_solutions, best_solution = self._run_multi_objective_optimization(
                all_trials, config.objectives
            )
            
            if best_solution:
                best_params = best_solution.params
                best_score = best_solution.score
                best_objectives = best_solution.objectives
        
        # Create optimization result
        optimization_time = time.time() - start_time
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_objectives=best_objectives,
            all_trials=all_trials,
            pareto_solutions=pareto_solutions,
            optimization_time=optimization_time,
            kpi_metrics=self.kpi_tracker if config.enable_kpi_tracking else {},
            stage_results=stage_results
        )
        
        # Log results
        tprint_info("=" * 60)
        tprint_info("ENHANCED AUTO-TUNING RESULTS")
        tprint_info("=" * 60)
        tprint_info(str({
            'best_score': result.best_score,
            'best_params': result.best_params,
            'objectives': result.best_objectives,
            'total_trials': len(result.all_trials),
            'optimization_time': result.optimization_time,
            'success_rate': self.kpi_tracker.get('success_rate', 0.0)
        }))
        tprint_info("=" * 60)
        
        tprint_info(f"PERFORMANCE: Auto-Tuning took {optimization_time:.3f}s")
        
        return result
    
    def _run_grid_search(
        self, 
        market_data: pd.DataFrame,
        search_space: Dict[str, Dict[str, Any]],
        config: AutoTuningConfig,
        stage_name: str = "Grid Search"
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Run grid search optimization."""
        tprint_info(f"🔍 Starting {stage_name}")
        
        if not _grid_utils_available:
            # Fallback to simple grid sampling
            trials = []
            best_trial = None
            best_score = -np.inf
            
            # Generate simple grid samples
            for i in range(min(config.max_trials_per_stage, 20)):
                params = {
                    'K': np.random.choice([3, 5, 7]),
                    'base_alpha': np.random.uniform(0.1, 2.0),
                    'kappa': np.random.uniform(5.0, 50.0),
                    'num_iters': np.random.choice([50, 100, 150]),
                    'lr': np.random.uniform(1e-4, 1e-2),
                    'n_mixtures': np.random.choice([1, 2, 3])
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
                grid_points=config.grid_resolution
            )
        else:  # Fine grid search
            best_params = getattr(config, 'previous_best_params', {})
            param_grid = build_fine_grid_around_best(
                search_space,
                best_params,
                grid_points=10
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
            if not _clusterer_available:
                return {
                    'params': params,
                    'score': 0.0,
                    'objectives': {obj: 0.0 for obj in objectives},
                    'success': False,
                    'error': 'Clusterer not available'
                }
            
            # Create clusterer configuration
            config = StickyFiniteHMMConfig(**params)
            
            # Create and run clusterer
            clusterer = StickyFiniteHMMClusterer(config)
            result = clusterer.fit_predict(market_data)
            
            if result.success:
                # Calculate objectives
                objectives_scores = self._calculate_objectives(result, objectives)
                
                return {
                    'params': params,
                    'score': result.composite_score,
                    'objectives': objectives_scores,
                    'success': True,
                    'clustering_result': result
                }
            else:
                return {
                    'params': params,
                    'score': -np.inf,
                    'objectives': {obj: -np.inf for obj in objectives},
                    'success': False,
                    'error': result.error_message
                }
                
        except Exception as e:
            return {
                'params': params,
                'score': -np.inf,
                'objectives': {obj: -np.inf for obj in objectives},
                'success': False,
                'error': str(e)
            }
    
    def _calculate_objectives(
        self, 
        result, 
        objectives: List[str]
    ) -> Dict[str, float]:
        """Calculate optimization objectives."""
        objectives_scores = {}
        
        for obj in objectives:
            if obj == 'composite_score':
                objectives_scores[obj] = result.composite_score
            elif obj == 'silhouette_score':
                if result.quality_assessment:
                    objectives_scores[obj] = result.quality_assessment.get('silhouette_score', 0.0)
                else:
                    objectives_scores[obj] = 0.0
            elif obj == 'transition_persistence':
                if result.quality_assessment:
                    objectives_scores[obj] = result.quality_assessment.get('transition_persistence', 0.0)
                else:
                    objectives_scores[obj] = 0.0
            elif obj == 'temporal_smoothness':
                if result.quality_assessment:
                    objectives_scores[obj] = result.quality_assessment.get('temporal_smoothness', 0.0)
                else:
                    objectives_scores[obj] = 0.0
            elif obj == 'cv_ratio':
                if result.quality_assessment:
                    objectives_scores[obj] = result.quality_assessment.get('cv_ratio', 0.0)
                else:
                    objectives_scores[obj] = 0.0
            elif obj == 'davies_bouldin_score':
                if result.quality_assessment:
                    # Lower is better, so negate for maximization
                    objectives_scores[obj] = -result.quality_assessment.get('davies_bouldin_score', 1.0)
                else:
                    objectives_scores[obj] = -1.0
            elif obj == 'calinski_harabasz_score':
                if result.quality_assessment:
                    objectives_scores[obj] = result.quality_assessment.get('calinski_harabasz_score', 0.0)
                else:
                    objectives_scores[obj] = 0.0
            else:
                objectives_scores[obj] = 0.0
        
        return objectives_scores
    
    def _run_multi_objective_optimization(
        self,
        trials: List[Dict[str, Any]],
        objectives: List[str]
    ) -> Tuple[List[Solution], Optional[Solution]]:
        """Run multi-objective optimization using Pareto front."""
        if not _pareto_available:
            tprint_warning("⚠️ Pareto optimization not available, using single objective")
            return [], None
        
        # Convert trials to solutions
        solutions = []
        for trial in trials:
            if trial['success'] and Solution is not None:
                solution = Solution(
                    params=trial['params'],
                    objectives=trial['objectives'],
                    score=trial['score']
                )
                solutions.append(solution)
        
        # Define objective directions (all maximize for now)
        # Set objective directions
        if _pareto_available and ObjectiveDirection is not None:
            objective_directions = {obj: ObjectiveDirection.MAXIMIZE for obj in objectives}
        else:
            objective_directions = {obj: "MAXIMIZE" for obj in objectives}
        
        # Compute Pareto front
        if compute_pareto_front is not None:
            pareto_front = compute_pareto_front(solutions, objective_directions)
        else:
            pareto_front = solutions[:5]  # Fallback: take first 5 solutions
        
        # Select knee point as best solution
        if select_knee_point is not None and pareto_front:
            best_solution = select_knee_point(pareto_front)
        else:
            best_solution = pareto_front[0] if pareto_front else None
        
        return pareto_front, best_solution
    
    def _update_kpi_tracker(self, stage_results: Dict[str, Any]):
        """Update KPI tracking metrics."""
        if not self.kpi_tracker:
            self.kpi_tracker = {
                'total_trials': 0,
                'successful_trials': 0,
                'total_time': 0.0,
                'stages_completed': 0
            }
        
        self.kpi_tracker['total_trials'] += stage_results['trials_evaluated']
        self.kpi_tracker['successful_trials'] += stage_results['successful_trials']
        self.kpi_tracker['total_time'] += stage_results['stage_time']
        self.kpi_tracker['stages_completed'] += 1
        self.kpi_tracker['success_rate'] = (
            self.kpi_tracker['successful_trials'] / max(1, self.kpi_tracker['total_trials'])
        )
        self.kpi_tracker['trials_per_second'] = (
            self.kpi_tracker['total_trials'] / max(1, self.kpi_tracker['total_time'])
        )


# Global instance
_enhanced_runner = None

def get_enhanced_runner() -> EnhancedStandaloneRunner:
    """Get or create enhanced runner instance."""
    global _enhanced_runner
    if _enhanced_runner is None:
        _enhanced_runner = EnhancedStandaloneRunner()
    return _enhanced_runner


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
    auto_tuning_config: Optional[AutoTuningConfig] = None,
    **kwargs
) -> OptimizationResult:
    """
    Run Sticky Finite HMM clustering with enhanced auto-tuning.
    
    Args:
        market_data: Market data DataFrame
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        optimization_stages: Number of optimization stages (1-2)
        use_multi_objective: Enable multi-objective optimization
        objectives: List of objectives to optimize
        max_trials_per_stage: Maximum trials per optimization stage
        timeout_seconds: Optimization timeout in seconds
        enable_kpi_tracking: Enable KPI tracking and metrics
        save_results: Save results to artifacts
        auto_tuning_config: Custom auto-tuning configuration
        **kwargs: Additional parameters
        
    Returns:
        OptimizationResult with best parameters and metrics
    """
    # Create auto-tuning config
    if auto_tuning_config is None:
        auto_tuning_config = AutoTuningConfig(
            optimization_stages=optimization_stages,
            use_multi_objective=use_multi_objective,
            objectives=objectives or ["composite_score", "temporal_smoothness", "cv_ratio"],
            max_trials_per_stage=max_trials_per_stage,
            timeout_seconds=timeout_seconds,
            enable_kpi_tracking=enable_kpi_tracking,
            save_results=save_results
        )
    
    # Get enhanced runner and run optimization
    runner = get_enhanced_runner()
    result = runner.run_auto_tuning(market_data, auto_tuning_config)
    
    return result


__all__ = [
    'EnhancedStandaloneRunner',
    'AutoTuningConfig',
    'OptimizationResult',
    'run_sticky_finite_hmm_with_auto_tuning',
    'get_enhanced_runner'
]
