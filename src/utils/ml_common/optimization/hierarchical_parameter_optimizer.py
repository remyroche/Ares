"""
Hierarchical Parameter Optimization Module

A general-purpose hierarchical optimization framework that allows efficient hyperparameter
tuning without needing to tune all parameters simultaneously. This addresses the curse of
dimensionality in hyperparameter optimization.

Key Features:
- Parameter Grouping: Organize parameters into logical groups (e.g., model structure, 
  regularization, learning rate) and optimize them sequentially or hierarchically
- Staged Optimization: Coarse grid → Fine grid → Advanced methods (TPE, BOHB)
- Backend Agnostic: Compatible with Optuna TPE, BOHB, Random Search, etc.
- Memory Efficient: Reduces search space by optimizing parameter groups independently
- Integration Ready: Works with existing optimization tools in this codebase

Example Usage:
    ```python
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
        HierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage
    )
    
    # Define parameter groups
    param_groups = [
        ParameterGroup(
            name="structure",
            params={
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 12}
            },
            priority=1  # Optimize first
        ),
        ParameterGroup(
            name="regularization",
            params={
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
                "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0}
            },
            priority=2,  # Optimize second
            depends_on=["structure"]  # After structure is optimized
        )
    ]
    
    # Create optimizer
    optimizer = HierarchicalParameterOptimizer(
        model=model,
        param_groups=param_groups,
        objective_func=objective_function,
        stages=[
            OptimizationStage.COARSE_GRID,
            OptimizationStage.FINE_GRID,
            OptimizationStage.TPE
        ]
    )
    
    # Run optimization
    best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
    ```

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │         Hierarchical Parameter Optimizer            │
    ├─────────────────────────────────────────────────────┤
    │  1. Parameter Grouping                              │
    │     - Group parameters by purpose/priority          │
    │     - Define dependencies between groups            │
    │                                                      │
    │  2. Sequential Group Optimization                   │
    │     - Optimize high-priority groups first           │
    │     - Fix optimized params, move to next group      │
    │                                                      │
    │  3. Staged Optimization (per group)                 │
    │     ┌─────────────────────────────────────┐        │
    │     │ Stage 1: Coarse Grid Search         │        │
    │     │   - Broad exploration (3-5 points)  │        │
    │     │   - Fast, rough best region         │        │
    │     └─────────────────────────────────────┘        │
    │                    ↓                                │
    │     ┌─────────────────────────────────────┐        │
    │     │ Stage 2: Fine Grid Search           │        │
    │     │   - Around best coarse region       │        │
    │     │   - Denser sampling (5-7 points)    │        │
    │     └─────────────────────────────────────┘        │
    │                    ↓                                │
    │     ┌─────────────────────────────────────┐        │
    │     │ Stage 3: Advanced Optimization      │        │
    │     │   - TPE: Tree Parzen Estimator     │        │
    │     │   - BOHB: Bayesian Opt + HyperBand │        │
    │     │   - Narrow search space            │        │
    │     └─────────────────────────────────────┘        │
    │                                                      │
    │  4. Final Refinement (optional)                     │
    │     - Joint optimization of all groups              │
    │     - Small refinement around best point            │
    └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from pathlib import Path
import json
from copy import deepcopy

from src.utils.logger import system_logger
from .grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)

from .execution_mode_adapter import adjust_hpo_params_for_mode, get_execution_mode

# Import custom_balanced_score for default HPO scoring
try:
    from .shared_utils.evaluation_metrics import calculate_custom_balanced_score_for_hpo
    CUSTOM_BALANCED_SCORE_AVAILABLE = True
except ImportError:
    CUSTOM_BALANCED_SCORE_AVAILABLE = False
    calculate_custom_balanced_score_for_hpo = None

# Optional imports
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = system_logger.getChild('HierarchicalParamOptimizer')


class OptimizationStage(Enum):
    """Optimization stages in hierarchical optimization."""
    COARSE_GRID = "coarse_grid"
    FINE_GRID = "fine_grid"
    TPE = "tpe"  # Tree-structured Parzen Estimator
    BOHB = "bohb"  # Bayesian Optimization HyperBand
    RANDOM = "random"  # Random search
    SMAC = "smac"  # Sequential Model-based Algorithm Configuration
    HYPEROPT = "hyperopt"  # HyperOpt TPE


class OptimizationBackend(Enum):
    """Supported optimization backends."""
    OPTUNA = "optuna"
    BOHB = "bohb"
    HYPEROPT = "hyperopt"
    SMAC = "smac"
    SKLEARN = "sklearn"


@dataclass
class ParameterGroup:
    """
    A group of related hyperparameters to be optimized together.
    
    Parameters can be grouped by:
    - Purpose (structure, regularization, learning)
    - Importance (critical, important, fine-tuning)
    - Computational cost (expensive, cheap)
    - Dependencies (parameter A affects optimal value of parameter B)
    """
    name: str
    params: Dict[str, Dict[str, Any]]  # Parameter name -> search space config
    priority: int = 1  # Lower number = higher priority (optimize first)
    depends_on: List[str] = field(default_factory=list)  # Names of groups that must be optimized first
    description: Optional[str] = None
    optimize_jointly: bool = False  # If True, optimize all params in group simultaneously
    
    def __post_init__(self):
        """Validate parameter group configuration."""
        if not self.params:
            raise ValueError(f"Parameter group '{self.name}' has no parameters")
        
        for param_name, param_config in self.params.items():
            if not isinstance(param_config, dict):
                raise ValueError(f"Parameter '{param_name}' config must be a dict")
            if 'type' not in param_config:
                raise ValueError(f"Parameter '{param_name}' must have a 'type' field")


@dataclass
class StageConfig:
    """Configuration for a single optimization stage."""
    stage: OptimizationStage
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    backend: OptimizationBackend = OptimizationBackend.OPTUNA
    
    # Grid search specific
    grid_points: int = 3  # Points per parameter for grid search
    
    # TPE/Bayesian specific
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    
    # BOHB specific
    min_budget: float = 1.0
    max_budget: float = 27.0
    eta: int = 3
    
    # General
    enable_pruning: bool = True
    random_state: int = 42


@dataclass
class OptimizationResult:
    """Result of optimization for a parameter group."""
    group_name: str
    stage: OptimizationStage
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    convergence_info: Optional[Dict[str, Any]] = None


@dataclass
class HierarchicalOptimizationResult:
    """Complete result of hierarchical optimization."""
    best_params: Dict[str, Any]  # Combined best parameters from all groups
    best_score: float
    group_results: List[OptimizationResult]  # Results for each parameter group
    total_time: float
    total_trials: int
    final_refinement_result: Optional[OptimizationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_time': self.total_time,
            'total_trials': self.total_trials,
            'group_results': [
                {
                    'group_name': r.group_name,
                    'stage': r.stage.value,
                    'best_params': r.best_params,
                    'best_score': r.best_score,
                    'n_trials': r.n_trials,
                    'optimization_time': r.optimization_time
                }
                for r in self.group_results
            ],
            'final_refinement': {
                'best_params': self.final_refinement_result.best_params,
                'best_score': self.final_refinement_result.best_score,
                'optimization_time': self.final_refinement_result.optimization_time
            } if self.final_refinement_result else None
        }


class HierarchicalParameterOptimizer:
    """
    Hierarchical parameter optimizer that optimizes parameter groups sequentially
    and applies staged optimization (coarse -> fine -> advanced) to each group.
    """
    
    def __init__(
        self,
        param_groups: List[ParameterGroup],
        objective_func: Callable,
        stages: Optional[List[OptimizationStage]] = None,
        stage_configs: Optional[Dict[OptimizationStage, StageConfig]] = None,
        cv_folds: int = 5,
        scoring_metric: str = 'custom_balanced_score',
        direction: str = 'maximize',
        n_rounds: int = 2,
        enable_final_refinement: bool = True,
        final_refinement_trials: int = 50,
        cache_dir: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = True,
        use_custom_balanced_score: bool = True
    ):
        """
        Initialize hierarchical parameter optimizer.
        
        Args:
            param_groups: List of parameter groups to optimize
            objective_func: Objective function(model, params, X, y) -> score
            stages: Optimization stages to use (default: [COARSE_GRID, FINE_GRID, TPE])
            stage_configs: Custom configuration for each stage
            cv_folds: Number of cross-validation folds
            scoring_metric: Metric to optimize (default: 'custom_balanced_score')
                           For ML trading models, 'custom_balanced_score' is recommended
                           as it balances financial performance, statistical accuracy,
                           regime awareness, and economic viability
            direction: 'maximize' or 'minimize'
            n_rounds: Number of rounds to iterate through all parameter groups (default: 2)
                     Round 1: Full exploration with coarse/fine/TPE
                     Round 2+: Refinement with narrowed search spaces
            enable_final_refinement: Whether to do final joint optimization
            final_refinement_trials: Number of trials for final refinement
            cache_dir: Directory to cache results
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
            use_custom_balanced_score: If True and scoring_metric='custom_balanced_score',
                                      uses the enhanced custom_balanced_score from evaluation_metrics.py
                                      (default: True)
        """
        self.param_groups = self._sort_param_groups_by_priority(param_groups)
        self.objective_func = objective_func
        self.stages = stages or [
            OptimizationStage.COARSE_GRID,
            OptimizationStage.FINE_GRID,
            OptimizationStage.TPE
        ]
        self.stage_configs = stage_configs or self._create_default_stage_configs()
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.direction = direction
        self.n_rounds = max(1, n_rounds)  # Ensure at least 1 round
        self.enable_final_refinement = enable_final_refinement
        self.final_refinement_trials = final_refinement_trials
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.verbose = verbose
        self.use_custom_balanced_score = use_custom_balanced_score
        
        self.execution_mode = get_execution_mode()

        # Adjust cv_folds and final_refinement_trials
        self.final_refinement_trials, self.cv_folds = adjust_hpo_params_for_mode(
            n_trials=self.final_refinement_trials,
            cv_folds=self.cv_folds,
            execution_mode=self.execution_mode
        )

        if self.execution_mode != 'full':
            logger.info(f"⚡ Mode {self.execution_mode.upper()}: cv_folds={self.cv_folds}, final_refinement_trials={self.final_refinement_trials}")

        # Adjust the stage_configs
        self.stage_configs = stage_configs or self._create_default_stage_configs()
        self._adjust_stage_configs() # Call new method to adjust configs
        
        # Internal state
        self.optimized_params: Dict[str, Any] = {}  # Accumulated best parameters
        self.group_results: List[OptimizationResult] = []
        self.round_results: List[Dict[str, Any]] = []  # Results for each round
        
        # Validate
        self._validate_configuration()
        
        # Setup
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('Optimizer')
        
        if verbose:
            self._print_configuration()
    
    def _sort_param_groups_by_priority(self, groups: List[ParameterGroup]) -> List[ParameterGroup]:
        """Sort parameter groups by priority and dependency order."""
        # First sort by priority
        sorted_groups = sorted(groups, key=lambda g: g.priority)
        
        # Then verify dependencies are satisfied
        optimized_group_names = set()
        final_order = []
        
        while sorted_groups:
            # Find next group whose dependencies are satisfied
            for i, group in enumerate(sorted_groups):
                if all(dep in optimized_group_names for dep in group.depends_on):
                    final_order.append(group)
                    optimized_group_names.add(group.name)
                    sorted_groups.pop(i)
                    break
            else:
                # No group found with satisfied dependencies
                remaining_names = [g.name for g in sorted_groups]
                raise ValueError(
                    f"Circular or unsatisfiable dependencies detected in parameter groups: {remaining_names}"
                )
        
        return final_order
    
    def _create_default_stage_configs(self) -> Dict[OptimizationStage, StageConfig]:
        """Create default configuration for each optimization stage."""
        return {
            OptimizationStage.COARSE_GRID: StageConfig(
                stage=OptimizationStage.COARSE_GRID,
                n_trials=50,
                grid_points=3,
                enable_pruning=False
            ),
            OptimizationStage.FINE_GRID: StageConfig(
                stage=OptimizationStage.FINE_GRID,
                n_trials=50,
                grid_points=5,
                enable_pruning=False
            ),
            OptimizationStage.TPE: StageConfig(
                stage=OptimizationStage.TPE,
                n_trials=100,
                n_startup_trials=10,
                n_ei_candidates=24,
                enable_pruning=True
            ),
            OptimizationStage.BOHB: StageConfig(
                stage=OptimizationStage.BOHB,
                n_trials=100,
                min_budget=1.0,
                max_budget=27.0,
                eta=3,
                enable_pruning=True
            ),
            OptimizationStage.RANDOM: StageConfig(
                stage=OptimizationStage.RANDOM,
                n_trials=50,
                enable_pruning=False
            )
        }
    
    def _validate_configuration(self):
        """Validate optimizer configuration."""
        if not self.param_groups:
            raise ValueError("At least one parameter group must be specified")
        
        if not self.stages:
            raise ValueError("At least one optimization stage must be specified")
        
        # Check that we have configs for all stages
        for stage in self.stages:
            if stage not in self.stage_configs:
                self.logger.warning(f"No config for stage {stage}, using defaults")
                self.stage_configs[stage] = self._create_default_stage_configs()[stage]
        
        # Validate dependencies
        group_names = {g.name for g in self.param_groups}
        for group in self.param_groups:
            for dep in group.depends_on:
                if dep not in group_names:
                    raise ValueError(
                        f"Parameter group '{group.name}' depends on unknown group '{dep}'"
                    )
    
    def _print_configuration(self):
        """Print optimizer configuration."""
        logger.info("=" * 80)
        logger.info("Hierarchical Parameter Optimizer Configuration")
        logger.info("=" * 80)
        logger.info(f"Number of parameter groups: {len(self.param_groups)}")
        logger.info(f"Optimization rounds: {self.n_rounds}")
        logger.info(f"Optimization stages: {[s.value for s in self.stages]}")
        logger.info(f"Direction: {self.direction}")
        logger.info(f"CV folds: {self.cv_folds}")
        logger.info(f"Scoring metric: {self.scoring_metric}")
        logger.info(f"Final refinement: {self.enable_final_refinement}")
        logger.info("")
        logger.info("Parameter Groups (in optimization order):")
        for i, group in enumerate(self.param_groups):
            logger.info(f"  {i+1}. {group.name} (priority={group.priority})")
            logger.info(f"     Parameters: {list(group.params.keys())}")
            if group.depends_on:
                logger.info(f"     Depends on: {group.depends_on}")
        logger.info("=" * 80)


    def _adjust_stage_configs(self):
        """Adjust n_trials and grid_points in stage configs based on execution mode."""
        if self.execution_mode == 'full':
            return
    
        logger.info(f"⚡ Mode {self.execution_mode.upper()}: Adjusting HPO stage configurations...")
        for stage, config in self.stage_configs.items():
    
            # Use grid_points as a base for cv_folds adjustment
            base_folds = config.grid_points if stage in [OptimizationStage.COARSE_GRID, OptimizationStage.FINE_GRID] else 5 
    
            adjusted_trials, adjusted_folds = adjust_hpo_params_for_mode(
                n_trials=config.n_trials,
                cv_folds=base_folds,
                execution_mode=self.execution_mode
            )
    
            if config.n_trials != adjusted_trials:
                logger.info(f"   Stage {stage.value}: n_trials adjusted from {config.n_trials} -> {adjusted_trials}")
                config.n_trials = adjusted_trials
    
            if stage in [OptimizationStage.COARSE_GRID, OptimizationStage.FINE_GRID]:
                if config.grid_points != adjusted_folds:
                    logger.info(f"   Stage {stage.value}: grid_points adjusted from {config.grid_points} -> {adjusted_folds}")
                    config.grid_points = adjusted_folds
    
            # Adjust n_startup_trials proportionally
            if hasattr(config, 'n_startup_trials') and config.n_trials > 0:
                original_trials = self._create_default_stage_configs()[stage].n_trials
                ratio = adjusted_trials / original_trials if original_trials > 0 else 0
                config.n_startup_trials = max(1, int(config.n_startup_trials * ratio))
  
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        initial_params: Optional[Dict[str, Any]] = None
    ) -> HierarchicalOptimizationResult:
        """
        Run hierarchical optimization with multiple rounds.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for holdout validation)
            y_val: Validation targets (optional)
            model: Model instance (if objective_func requires it)
            initial_params: Initial parameter values to start from
        
        Returns:
            HierarchicalOptimizationResult with best parameters and details
        """
        start_time = time.time()
        
        logger.info("🚀 Starting hierarchical parameter optimization")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Features: {X_train.shape[1] if hasattr(X_train, 'shape') else 'N/A'}")
        logger.info(f"   Number of rounds: {self.n_rounds}")
        
        # Initialize with any provided initial parameters
        if initial_params:
            self.optimized_params = deepcopy(initial_params)
            logger.info(f"   Starting with {len(initial_params)} initial parameters")
        
        # Optimize through multiple rounds
        total_trials = 0
        best_score_overall = float('-inf') if self.direction == 'maximize' else float('inf')
        
        for round_num in range(1, self.n_rounds + 1):
            logger.info("")
            logger.info("█" * 80)
            logger.info(f"🔄 ROUND {round_num}/{self.n_rounds}")
            logger.info("█" * 80)
            
            round_start_time = time.time()
            round_group_results = []
            round_start_score = best_score_overall
            
            # Determine if this is a refinement round
            is_refinement_round = round_num > 1
            
            # Optimize each parameter group sequentially
            for group_idx, group in enumerate(self.param_groups):
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"📊 Round {round_num} - Optimizing Group {group_idx + 1}/{len(self.param_groups)}: '{group.name}'")
                logger.info(f"   Priority: {group.priority}")
                logger.info(f"   Parameters: {list(group.params.keys())}")
                if is_refinement_round:
                    logger.info(f"   Mode: Refinement (narrowed search space)")
                else:
                    logger.info(f"   Mode: Exploration (full search space)")
                logger.info("=" * 80)
                
                # For refinement rounds, use narrowed search space around current best
                if is_refinement_round:
                    # Create narrowed search space for this group
                    group_best_params = {k: v for k, v in self.optimized_params.items() if k in group.params}
                    narrowed_group = self._create_narrowed_group(group, group_best_params)
                    group_to_optimize = narrowed_group
                else:
                    group_to_optimize = group
                
                # Optimize this group through all stages
                group_result = self._optimize_parameter_group(
                    group=group_to_optimize,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model=model,
                    fixed_params=self.optimized_params.copy(),
                    round_num=round_num,
                    is_refinement=is_refinement_round
                )
                
                # Update optimized parameters with results from this group
                self.optimized_params.update(group_result.best_params)
                round_group_results.append(group_result)
                self.group_results.append(group_result)
                total_trials += group_result.n_trials
                
                # Update best score
                is_better = (
                    (self.direction == 'maximize' and group_result.best_score > best_score_overall) or
                    (self.direction == 'minimize' and group_result.best_score < best_score_overall)
                )
                if is_better or round_num == 1:
                    best_score_overall = group_result.best_score
                
                logger.info(f"✅ Group '{group.name}' optimization complete")
                logger.info(f"   Best score: {group_result.best_score:.6f}")
                logger.info(f"   Best params: {group_result.best_params}")
                logger.info(f"   Time: {group_result.optimization_time:.2f}s")
            
            # Round summary
            round_time = time.time() - round_start_time
            round_improvement = best_score_overall - round_start_score if round_num > 1 else 0.0
            
            self.round_results.append({
                'round': round_num,
                'best_score': best_score_overall,
                'improvement': round_improvement,
                'time': round_time,
                'trials': sum(r.n_trials for r in round_group_results),
                'group_results': round_group_results
            })
            
            logger.info("")
            logger.info("─" * 80)
            logger.info(f"✅ Round {round_num} Complete")
            logger.info(f"   Round best score: {best_score_overall:.6f}")
            if round_num > 1:
                logger.info(f"   Improvement from previous: {round_improvement:+.6f}")
            logger.info(f"   Round time: {round_time:.2f}s")
            logger.info(f"   Round trials: {sum(r.n_trials for r in round_group_results)}")
            logger.info("─" * 80)
        
        # Final refinement: jointly optimize all parameters around best point
        final_refinement_result = None
        if self.enable_final_refinement:
            logger.info("")
            logger.info("=" * 80)
            logger.info("🔧 Final Refinement: Joint optimization of all parameters")
            logger.info("=" * 80)
            
            final_refinement_result = self._final_refinement(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model=model,
                current_best_params=self.optimized_params.copy()
            )
            
            is_better = (
                (self.direction == 'maximize' and final_refinement_result.best_score > best_score_overall) or
                (self.direction == 'minimize' and final_refinement_result.best_score < best_score_overall)
            )
            
            if is_better:
                logger.info(f"✅ Final refinement improved score from {best_score_overall:.6f} to {final_refinement_result.best_score:.6f}")
                self.optimized_params = final_refinement_result.best_params
                best_score_overall = final_refinement_result.best_score
            else:
                logger.info(f"ℹ️  Final refinement did not improve score")
            
            total_trials += final_refinement_result.n_trials
        
        total_time = time.time() - start_time
        
        # Create final result
        result = HierarchicalOptimizationResult(
            best_params=self.optimized_params,
            best_score=best_score_overall,
            group_results=self.group_results,
            total_time=total_time,
            total_trials=total_trials,
            final_refinement_result=final_refinement_result
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎉 Hierarchical Optimization Complete!")
        logger.info("=" * 80)
        logger.info(f"   Rounds completed: {self.n_rounds}")
        logger.info(f"   Best score: {result.best_score:.6f}")
        logger.info(f"   Total time: {result.total_time:.2f}s")
        logger.info(f"   Total trials: {result.total_trials}")
        logger.info(f"   Best parameters: {result.best_params}")
        logger.info("")
        logger.info("   Round-by-round summary:")
        for round_info in self.round_results:
            improvement_str = f" (improvement: {round_info['improvement']:+.6f})" if round_info['round'] > 1 else ""
            logger.info(f"     Round {round_info['round']}: score={round_info['best_score']:.6f}{improvement_str}")
        logger.info("=" * 80)
        
        # Save results if cache directory is specified
        if self.cache_dir:
            self._save_results(result)
        
        return result
    
    def _optimize_parameter_group(
        self,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        round_num: int = 1,
        is_refinement: bool = False
    ) -> OptimizationResult:
        """
        Optimize a single parameter group through all stages.
        
        Args:
            group: Parameter group to optimize
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            model: Model instance
            fixed_params: Parameters that are already optimized (fixed)
            round_num: Current optimization round number
            is_refinement: Whether this is a refinement round (narrowed search space)
        
        Returns:
            OptimizationResult for this group
        """
        group_start_time = time.time()
        
        current_best_params = {}
        current_best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        all_trials = []
        total_group_trials = 0
        
        # Run through each optimization stage
        for stage_idx, stage in enumerate(self.stages):
            logger.info("")
            logger.info(f"  Stage {stage_idx + 1}/{len(self.stages)}: {stage.value}")
            logger.info(f"  {'─' * 76}")
            
            stage_result = self._optimize_stage(
                stage=stage,
                group=group,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model=model,
                fixed_params=fixed_params,
                current_best_params=current_best_params.copy()
            )
            
            all_trials.extend(stage_result.all_trials)
            total_group_trials += stage_result.n_trials
            
            # Update best if this stage improved
            is_better = (
                (self.direction == 'maximize' and stage_result.best_score > current_best_score) or
                (self.direction == 'minimize' and stage_result.best_score < current_best_score)
            )
            
            if is_better or not current_best_params:
                current_best_score = stage_result.best_score
                current_best_params = stage_result.best_params.copy()
                logger.info(f"  ✅ {stage.value} improved score to {current_best_score:.6f}")
            else:
                logger.info(f"  ℹ️  {stage.value} did not improve score (keeping {current_best_score:.6f})")
        
        group_time = time.time() - group_start_time
        
        return OptimizationResult(
            group_name=group.name,
            stage=self.stages[-1],  # Final stage
            best_params=current_best_params,
            best_score=current_best_score,
            n_trials=total_group_trials,
            optimization_time=group_time,
            all_trials=all_trials
        )
    
    def _optimize_stage(
        self,
        stage: OptimizationStage,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        current_best_params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize a parameter group in a single stage.
        
        Args:
            stage: Optimization stage to run
            group: Parameter group
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model: Model instance
            fixed_params: Fixed parameters from previous groups
            current_best_params: Best parameters from previous stages of this group
        
        Returns:
            OptimizationResult for this stage
        """
        stage_config = self.stage_configs[stage]
        stage_start_time = time.time()
        
        if stage == OptimizationStage.COARSE_GRID:
            result = self._coarse_grid_search(
                group, X_train, y_train, X_val, y_val, model, fixed_params, stage_config
            )
        elif stage == OptimizationStage.FINE_GRID:
            result = self._fine_grid_search(
                group, X_train, y_train, X_val, y_val, model, fixed_params,
                current_best_params, stage_config
            )
        elif stage == OptimizationStage.TPE:
            result = self._tpe_optimization(
                group, X_train, y_train, X_val, y_val, model, fixed_params,
                current_best_params, stage_config
            )
        elif stage == OptimizationStage.BOHB:
            result = self._bohb_optimization(
                group, X_train, y_train, X_val, y_val, model, fixed_params,
                current_best_params, stage_config
            )
        elif stage == OptimizationStage.RANDOM:
            result = self._random_search(
                group, X_train, y_train, X_val, y_val, model, fixed_params, stage_config
            )
        else:
            raise ValueError(f"Unsupported optimization stage: {stage}")
        
        result.optimization_time = time.time() - stage_start_time
        return result
    
    def _coarse_grid_search(
        self,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        stage_config: StageConfig
    ) -> OptimizationResult:
        """Perform coarse grid search."""
        logger.info(f"    Running coarse grid search ({stage_config.grid_points} points per param)")
        
        # Build coarse grid
        grid = build_coarse_grid_from_search_space(
            group.params,
            grid_points=stage_config.grid_points
        )
        
        if not grid:
            logger.warning("    Empty grid generated, using fallback")
            grid = [self._sample_random_params(group.params)]
        
        logger.info(f"    Grid size: {len(grid)} combinations")
        
        # Evaluate each combination
        best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        best_params = {}
        all_trials = []
        
        for i, params in enumerate(grid):
            # Merge with fixed params
            full_params = {**fixed_params, **params}
            
            # Evaluate
            score = self._evaluate_params(
                full_params, X_train, y_train, X_val, y_val, model
            )
            
            all_trials.append({
                'params': params,
                'score': score,
                'trial_number': i
            })
            
            # Update best
            is_better = (
                (self.direction == 'maximize' and score > best_score) or
                (self.direction == 'minimize' and score < best_score)
            )
            
            if is_better:
                best_score = score
                best_params = params.copy()
            
            if (i + 1) % 10 == 0:
                logger.debug(f"    Evaluated {i + 1}/{len(grid)} combinations")
        
        logger.info(f"    Coarse grid complete. Best score: {best_score:.6f}")
        
        return OptimizationResult(
            group_name=group.name,
            stage=OptimizationStage.COARSE_GRID,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(grid),
            optimization_time=0.0,  # Set by caller
            all_trials=all_trials
        )
    
    def _fine_grid_search(
        self,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        current_best_params: Dict[str, Any],
        stage_config: StageConfig
    ) -> OptimizationResult:
        """Perform fine grid search around current best parameters."""
        logger.info(f"    Running fine grid search around best coarse parameters")
        
        if not current_best_params:
            logger.warning("    No best params from previous stage, using full coarse grid")
            return self._coarse_grid_search(
                group, X_train, y_train, X_val, y_val, model, fixed_params, stage_config
            )
        
        # Build fine grid around best parameters
        grid = build_fine_grid_around_best(
            group.params,
            current_best_params,
            grid_points=stage_config.grid_points
        )
        
        if not grid:
            logger.warning("    Empty fine grid, keeping best params")
            return OptimizationResult(
                group_name=group.name,
                stage=OptimizationStage.FINE_GRID,
                best_params=current_best_params,
                best_score=float('-inf') if self.direction == 'maximize' else float('inf'),
                n_trials=0,
                optimization_time=0.0,
                all_trials=[]
            )
        
        logger.info(f"    Fine grid size: {len(grid)} combinations")
        
        # Evaluate each combination
        best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        best_params = current_best_params.copy()
        all_trials = []
        
        for i, params in enumerate(grid):
            # Merge with fixed params
            full_params = {**fixed_params, **params}
            
            # Evaluate
            score = self._evaluate_params(
                full_params, X_train, y_train, X_val, y_val, model
            )
            
            all_trials.append({
                'params': params,
                'score': score,
                'trial_number': i
            })
            
            # Update best
            is_better = (
                (self.direction == 'maximize' and score > best_score) or
                (self.direction == 'minimize' and score < best_score)
            )
            
            if is_better:
                best_score = score
                best_params = params.copy()
        
        logger.info(f"    Fine grid complete. Best score: {best_score:.6f}")
        
        return OptimizationResult(
            group_name=group.name,
            stage=OptimizationStage.FINE_GRID,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(grid),
            optimization_time=0.0,
            all_trials=all_trials
        )
    
    def _tpe_optimization(
        self,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        current_best_params: Dict[str, Any],
        stage_config: StageConfig
    ) -> OptimizationResult:
        """Perform TPE optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.warning("    Optuna not available, falling back to random search")
            return self._random_search(
                group, X_train, y_train, X_val, y_val, model, fixed_params, stage_config
            )
        
        logger.info(f"    Running TPE optimization ({stage_config.n_trials} trials)")
        
        # Narrow search space around current best (if available)
        search_space = self._create_narrowed_search_space(
            group.params, 
            current_best_params,
            narrow_factor=0.1,
            use_log_space_narrowing=True,  # Enable log-space narrowing
            importance_weights=None  # Not used in TPE stage (only in final refinement)
        ) if current_best_params else group.params
        
        # Create Optuna study
        direction = 'maximize' if self.direction == 'maximize' else 'minimize'
        sampler = TPESampler(
            n_startup_trials=stage_config.n_startup_trials,
            n_ei_candidates=stage_config.n_ei_candidates,
            seed=stage_config.random_state
        )
        pruner = MedianPruner() if stage_config.enable_pruning else None
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective
        def objective(trial):
            # Sample parameters
            params = self._sample_params_from_optuna(trial, search_space)
            
            # Merge with fixed params
            full_params = {**fixed_params, **params}
            
            # Evaluate
            score = self._evaluate_params(
                full_params, X_train, y_train, X_val, y_val, model
            )
            
            return score
        
        # Optimize
        study.optimize(
            objective,
            n_trials=stage_config.n_trials,
            timeout=stage_config.timeout_seconds,
            show_progress_bar=False
        )
        
        # Extract results
        best_trial = study.best_trial
        best_params = {k: v for k, v in best_trial.params.items() if k in group.params}
        best_score = best_trial.value
        
        all_trials = [
            {
                'params': {k: v for k, v in t.params.items() if k in group.params},
                'score': t.value,
                'trial_number': t.number
            }
            for t in study.trials
            if t.value is not None
        ]
        
        logger.info(f"    TPE optimization complete. Best score: {best_score:.6f}")
        logger.info(f"    Completed {len(study.trials)} trials")
        
        return OptimizationResult(
            group_name=group.name,
            stage=OptimizationStage.TPE,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            optimization_time=0.0,
            all_trials=all_trials
        )
    
    def _bohb_optimization(
        self,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        current_best_params: Dict[str, Any],
        stage_config: StageConfig
    ) -> OptimizationResult:
        """Perform BOHB optimization (Bayesian Optimization + HyperBand)."""
        # BOHB requires additional dependencies, fall back to TPE if not available
        logger.info("    BOHB optimization not yet implemented, falling back to TPE")
        return self._tpe_optimization(
            group, X_train, y_train, X_val, y_val, model, fixed_params,
            current_best_params, stage_config
        )
    
    def _random_search(
        self,
        group: ParameterGroup,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        fixed_params: Dict[str, Any],
        stage_config: StageConfig
    ) -> OptimizationResult:
        """Perform random search."""
        logger.info(f"    Running random search ({stage_config.n_trials} trials)")
        
        best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        best_params = {}
        all_trials = []
        
        for i in range(stage_config.n_trials):
            # Sample random parameters
            params = self._sample_random_params(group.params)
            
            # Merge with fixed params
            full_params = {**fixed_params, **params}
            
            # Evaluate
            score = self._evaluate_params(
                full_params, X_train, y_train, X_val, y_val, model
            )
            
            all_trials.append({
                'params': params,
                'score': score,
                'trial_number': i
            })
            
            # Update best
            is_better = (
                (self.direction == 'maximize' and score > best_score) or
                (self.direction == 'minimize' and score < best_score)
            )
            
            if is_better:
                best_score = score
                best_params = params.copy()
        
        logger.info(f"    Random search complete. Best score: {best_score:.6f}")
        
        return OptimizationResult(
            group_name=group.name,
            stage=OptimizationStage.RANDOM,
            best_params=best_params,
            best_score=best_score,
            n_trials=stage_config.n_trials,
            optimization_time=0.0,
            all_trials=all_trials
        )
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance from optimization history.
        
        Analyzes trial history to determine which parameters have the most impact
        on the objective score. Uses correlation-based sensitivity analysis.
        
        Returns:
            Dict mapping parameter names to importance scores [0, 1]
            Higher importance = more sensitive parameter = should narrow more
        """
        if not self.group_results:
            return {}
        
        logger.info("    📊 Analyzing parameter importance from trial history...")
        
        # Collect all trials across all groups
        all_trial_data = {}  # param_name -> [(value, score), ...]
        
        for group_result in self.group_results:
            for trial in group_result.all_trials:
                params = trial.get('params', {})
                score = trial.get('score', 0.0)
                
                for param_name, param_value in params.items():
                    if param_name not in all_trial_data:
                        all_trial_data[param_name] = []
                    
                    all_trial_data[param_name].append({
                        'value': param_value,
                        'score': score
                    })
        
        # Calculate importance (sensitivity) for each parameter
        importance = {}
        
        for param_name, data_points in all_trial_data.items():
            if len(data_points) < 3:  # Need at least 3 points for meaningful correlation
                importance[param_name] = 0.5  # Default medium importance
                continue
            
            try:
                values = np.array([d['value'] for d in data_points])
                scores = np.array([d['score'] for d in data_points])
                
                # Calculate correlation (absolute value - direction doesn't matter)
                if len(np.unique(values)) > 1:  # Need variation in parameter values
                    correlation = np.corrcoef(values, scores)[0, 1]
                    # Absolute correlation = sensitivity
                    sensitivity = abs(correlation) if not np.isnan(correlation) else 0.5
                    importance[param_name] = float(np.clip(sensitivity, 0.0, 1.0))
                else:
                    importance[param_name] = 0.5  # No variation, unknown importance
                
                logger.debug(f"      {param_name}: importance={importance[param_name]:.3f}")
            
            except Exception as e:
                logger.debug(f"      {param_name}: importance calculation failed ({e}), using default")
                importance[param_name] = 0.5
        
        if importance:
            logger.info(f"    ✅ Parameter importance calculated for {len(importance)} parameters")
            # Log top 3 most important parameters
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"       Most important: {sorted_importance[:3]}")
        
        return importance
    
    def _final_refinement(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any],
        current_best_params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Enhanced final refinement with adaptive parameter importance.
        
        Jointly optimize all parameters around the best point with:
        - Log-space narrowing for log-scale parameters
        - Adaptive narrowing based on parameter importance
        - Better handling of interaction effects
        """
        logger.info(f"    Running enhanced final joint refinement ({self.final_refinement_trials} trials)")
        logger.info(f"    Enhancements: Log-space narrowing + Adaptive importance weighting")
        
        # Combine all parameter groups into one
        all_params = {}
        for group in self.param_groups:
            all_params.update(group.params)
        
        logger.info(f"    Combined {len(all_params)} parameters from {len(self.param_groups)} groups")
        
        # Calculate parameter importance from trial history
        importance_weights = self._calculate_parameter_importance()
        
        if importance_weights:
            logger.info(f"    Adaptive narrowing enabled: important params narrowed more")
        else:
            logger.info(f"    Using uniform narrowing (no trial history available)")
        
        # Create adaptive narrowed search space
        # Important parameters get narrower ranges (more focus)
        # Less important parameters get wider ranges (more exploration)
        logger.info(f"    Creating adaptive narrowed search space...")
        narrow_space = self._create_narrowed_search_space(
            all_params, 
            current_best_params,
            narrow_factor=0.1,
            use_log_space_narrowing=True,  # Enable log-space narrowing
            importance_weights=importance_weights  # Adaptive based on importance
        )
        logger.info(f"    ✅ Narrowed search space created")
        
        if not OPTUNA_AVAILABLE:
            logger.warning("    Optuna not available for final refinement")
            return OptimizationResult(
                group_name="final_refinement",
                stage=OptimizationStage.TPE,
                best_params=current_best_params,
                best_score=float('-inf') if self.direction == 'maximize' else float('inf'),
                n_trials=0,
                optimization_time=0.0,
                all_trials=[]
            )
        
        # Create Optuna study
        direction = 'maximize' if self.direction == 'maximize' else 'minimize'
        sampler = TPESampler(
            n_startup_trials=min(10, self.final_refinement_trials // 5),
            n_ei_candidates=24,
            seed=self.random_state
        )
        
        study = optuna.create_study(direction=direction, sampler=sampler)
        
        # Define objective
        def objective(trial):
            params = self._sample_params_from_optuna(trial, narrow_space)
            score = self._evaluate_params(params, X_train, y_train, X_val, y_val, model)
            return score
        
        # Optimize
        study.optimize(objective, n_trials=self.final_refinement_trials, show_progress_bar=False)
        
        # Extract results
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_score = best_trial.value
        
        all_trials = [
            {'params': t.params, 'score': t.value, 'trial_number': t.number}
            for t in study.trials
            if t.value is not None
        ]
        
        logger.info(f"    Final refinement complete. Best score: {best_score:.6f}")
        
        return OptimizationResult(
            group_name="final_refinement",
            stage=OptimizationStage.TPE,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            optimization_time=0.0,
            all_trials=all_trials
        )
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model: Optional[Any]
    ) -> float:
        """Evaluate a set of parameters using the objective function."""
        try:
            score = self.objective_func(
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model=model,
                cv_folds=self.cv_folds,
                scoring_metric=self.scoring_metric
            )
            return score
        except Exception as e:
            logger.warning(f"    Evaluation failed: {e}")
            return float('-inf') if self.direction == 'maximize' else float('inf')
    
    def _sample_params_from_optuna(
        self,
        trial: 'optuna.Trial',
        search_space: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sample parameters from search space using Optuna trial."""
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
        
        return params
    
    def _sample_random_params(self, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        import random
        
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                low, high = param_config['low'], param_config['high']
                if param_config.get('log', False):
                    params[param_name] = np.exp(random.uniform(np.log(low), np.log(high)))
                else:
                    params[param_name] = random.uniform(low, high)
            elif param_type == 'int':
                params[param_name] = random.randint(param_config['low'], param_config['high'])
            elif param_type == 'categorical':
                params[param_name] = random.choice(param_config['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
        
        return params
    
    def _create_narrowed_search_space(
        self,
        search_space: Dict[str, Dict[str, Any]],
        best_params: Dict[str, Any],
        narrow_factor: float = 0.1,
        use_log_space_narrowing: bool = True,
        importance_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a narrowed search space around best parameters with adaptive scaling.
        
        Enhanced with:
        - Log-space narrowing for log-scale parameters
        - Adaptive narrowing based on parameter importance
        - Proper handling of different parameter scales
        
        Args:
            search_space: Original search space
            best_params: Best parameters found so far
            narrow_factor: Base factor to narrow range (0.1 = ±10% of original range)
            use_log_space_narrowing: If True, narrow log-scale params in log space
            importance_weights: Optional dict of parameter importance scores [0, 1]
                               Higher importance → narrower range (focus optimization)
                               Lower importance → wider range (allow exploration)
        
        Returns:
            Narrowed search space with adaptive scaling
        """
        narrowed = {}
        
        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                narrowed[param_name] = param_config.copy()
                continue
            
            best_value = best_params[param_name]
            narrowed_config = param_config.copy()
            
            # Calculate adaptive narrow factor based on parameter importance
            if importance_weights and param_name in importance_weights:
                importance = importance_weights[param_name]
                # High importance (close to 1.0) → narrow more (focus here)
                # Low importance (close to 0.0) → narrow less (explore more)
                adaptive_factor = narrow_factor * (0.5 + importance)
            else:
                adaptive_factor = narrow_factor
            
            if param_config['type'] == 'float':
                low, high = param_config['low'], param_config['high']
                
                # Enhanced: narrow in log space for log-scale parameters
                if use_log_space_narrowing and param_config.get('log', False):
                    # Log-space narrowing (proper for learning_rate, reg_alpha, etc.)
                    log_low = np.log(max(low, 1e-10))
                    log_high = np.log(max(high, 1e-10))
                    log_best = np.log(max(best_value, 1e-10))
                    log_range = log_high - log_low
                    
                    narrow_log_range = log_range * adaptive_factor
                    narrowed_log_low = max(log_low, log_best - narrow_log_range)
                    narrowed_log_high = min(log_high, log_best + narrow_log_range)
                    
                    # Convert back to linear space
                    narrowed_config['low'] = max(low, np.exp(narrowed_log_low))
                    narrowed_config['high'] = min(high, np.exp(narrowed_log_high))
                    
                    if self.verbose:
                        importance = importance_weights.get(param_name, 0.5) if importance_weights else 0.5
                        logger.debug(
                            f"      {param_name} (log-scale, importance={importance:.2f}): "
                            f"[{low:.6f}, {high:.6f}] → [{narrowed_config['low']:.6f}, {narrowed_config['high']:.6f}]"
                        )
                else:
                    # Linear narrowing (original approach)
                    range_size = high - low
                    narrow_range = range_size * adaptive_factor
                    
                    narrowed_config['low'] = max(low, best_value - narrow_range)
                    narrowed_config['high'] = min(high, best_value + narrow_range)
                    
                    if self.verbose:
                        importance = importance_weights.get(param_name, 0.5) if importance_weights else 0.5
                        logger.debug(
                            f"      {param_name} (linear, importance={importance:.2f}): "
                            f"[{low:.4f}, {high:.4f}] → [{narrowed_config['low']:.4f}, {narrowed_config['high']:.4f}]"
                        )
            
            elif param_config['type'] == 'int':
                low, high = param_config['low'], param_config['high']
                narrow_amount = max(1, int((high - low) * adaptive_factor))
                
                narrowed_config['low'] = max(low, best_value - narrow_amount)
                narrowed_config['high'] = min(high, best_value + narrow_amount)
                
                if self.verbose:
                    importance = importance_weights.get(param_name, 0.5) if importance_weights else 0.5
                    logger.debug(
                        f"      {param_name} (int, importance={importance:.2f}): "
                        f"[{low}, {high}] → [{narrowed_config['low']}, {narrowed_config['high']}]"
                    )
            
            # Categorical parameters stay the same
            narrowed[param_name] = narrowed_config
        
        return narrowed
    
    def _create_narrowed_group(
        self,
        group: ParameterGroup,
        best_params: Dict[str, Any],
        narrow_factor: float = 0.15
    ) -> ParameterGroup:
        """
        Create a narrowed parameter group around best parameters for refinement rounds.
        
        Enhanced with log-space narrowing for proper parameter scaling.
        
        Args:
            group: Original parameter group
            best_params: Best parameters found for this group
            narrow_factor: Factor to narrow range (0.15 = ±15% of original range)
        
        Returns:
            New ParameterGroup with narrowed search space
        """
        # For refinement rounds, we don't use importance weights (not enough history yet)
        # But we do use log-space narrowing
        narrowed_params = self._create_narrowed_search_space(
            group.params,
            best_params,
            narrow_factor=narrow_factor,
            use_log_space_narrowing=True,  # Enable log-space narrowing
            importance_weights=None  # Not used for group refinement
        )
        
        # Create new group with narrowed parameters but same metadata
        narrowed_group = ParameterGroup(
            name=group.name,
            params=narrowed_params,
            priority=group.priority,
            depends_on=group.depends_on,
            description=group.description,
            optimize_jointly=group.optimize_jointly
        )
        
        return narrowed_group
    
    def _save_results(self, result: HierarchicalOptimizationResult):
        """Save optimization results to cache directory."""
        if not self.cache_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.cache_dir) / f"hierarchical_opt_results_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info(f"💾 Results saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")


# Convenience functions

def create_param_group(
    name: str,
    params: Dict[str, Dict[str, Any]],
    priority: int = 1,
    depends_on: Optional[List[str]] = None,
    description: Optional[str] = None
) -> ParameterGroup:
    """
    Convenience function to create a parameter group.
    
    Example:
        structure_group = create_param_group(
            name="structure",
            params={
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 12}
            },
            priority=1,
            description="Model structure parameters"
        )
    """
    return ParameterGroup(
        name=name,
        params=params,
        priority=priority,
        depends_on=depends_on or [],
        description=description
    )


def default_objective_function(
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    cv_folds: int = 5,
    scoring_metric: str = 'neg_mean_squared_error'
) -> float:
    """
    Default objective function for optimization.
    
    Uses cross-validation if X_val is None, otherwise uses holdout validation.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for default objective function")
    
    if model is None:
        raise ValueError("Model must be provided to objective function")
    
    try:
        # Set parameters
        model.set_params(**params)
        
        # Use holdout validation if validation set is provided
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if scoring_metric == 'neg_mean_squared_error':
                return -mean_squared_error(y_val, y_pred)
            elif scoring_metric == 'neg_mean_absolute_error':
                return -mean_absolute_error(y_val, y_pred)
            elif scoring_metric == 'r2':
                return r2_score(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported scoring metric: {scoring_metric}")
        
        # Use cross-validation
        cv = TimeSeriesSplit(n_splits=cv_folds)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=1
        )
        
        return np.mean(scores)
    
    except Exception as e:
        logger.warning(f"Objective function failed: {e}")
        return float('-inf')


def create_custom_balanced_score_objective(
    model_trainer: Callable,
    use_returns: bool = True,
    use_regime_labels: bool = False
) -> Callable:
    """
    Create an objective function that uses custom_balanced_score for HPO.
    
    This is a convenience function to create objective functions compatible with
    HierarchicalParameterOptimizer that use the recommended custom_balanced_score.
    
    Args:
        model_trainer: Function(params, X_train, y_train, X_val, y_val) -> (model, predictions)
                      that trains a model and returns predictions
        use_returns: Whether to calculate returns from predictions for financial metrics
        use_regime_labels: Whether to use regime labels (if available in kwargs)
        
    Returns:
        Callable: Objective function compatible with HierarchicalParameterOptimizer
        
    Example:
        ```python
        def train_my_model(params, X_train, y_train, X_val, y_val):
            model = MyModel(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            return model, predictions
        
        objective_func = create_custom_balanced_score_objective(train_my_model)
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_func,
            direction='maximize'  # custom_balanced_score should be maximized
        )
        ```
    """
    if not CUSTOM_BALANCED_SCORE_AVAILABLE:
        logger.warning("custom_balanced_score not available, returning basic objective")
        return default_objective_function
    
    def objective_func(
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        cv_folds: int = 5,
        scoring_metric: str = 'custom_balanced_score',
        **kwargs
    ) -> float:
        """Objective function using custom_balanced_score."""
        try:
            # Train model and get predictions
            trained_model, predictions = model_trainer(params, X_train, y_train, X_val, y_val)
            
            # Calculate returns if requested
            returns = None
            if use_returns and y_val is not None:
                # Simple return calculation: pred * actual
                # More sophisticated return calculation can be provided in kwargs
                returns = kwargs.get('returns', predictions * y_val)
            
            # Get regime labels if requested
            regime_labels = None
            if use_regime_labels:
                regime_labels = kwargs.get('regime_labels', None)
            
            # Calculate custom_balanced_score
            score = calculate_custom_balanced_score_for_hpo(
                predictions=predictions,
                targets=y_val,
                returns=returns,
                regime_labels=regime_labels
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return 0.0  # Return poor score on failure
    
    return objective_func


__all__ = [
    'HierarchicalParameterOptimizer',
    'ParameterGroup',
    'OptimizationStage',
    'OptimizationBackend',
    'StageConfig',
    'OptimizationResult',
    'HierarchicalOptimizationResult',
    'create_param_group',
    'default_objective_function',
    'create_custom_balanced_score_objective',
    'CUSTOM_BALANCED_SCORE_AVAILABLE',
]
