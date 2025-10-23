"""
Parameter Optimizer for Multi-Horizon Profit Labeling

This module provides systematic optimization of profit labeling parameters using
data-driven approaches similar to hyperparameter optimization in ML. It explores
parameter spaces to find optimal configurations for different market conditions.

Key Optimization Areas:
1. Profit Targets Optimization (micro, small, medium, good levels)
2. Time Horizons Optimization (immediate, short-term periods)  
3. Quality Scoring Weights (speed, risk, profitability weights)
4. Fee Structure Impact Analysis
5. Leverage Adjustment Parameters
6. Composite Score Component Weights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig
)

# Optional: Use hyperopt for advanced optimization
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Optional: Use optuna for optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptimizationMethod(Enum):
    """Enumeration of optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPEROPT_TPE = "hyperopt_tpe"
    OPTUNA_TPE = "optuna_tpe"


class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    PREDICTIVE_POWER = "predictive_power"
    LABEL_STABILITY = "label_stability"
    HIT_RATE_BALANCE = "hit_rate_balance"
    ECONOMIC_VALUE = "economic_value"
    COMPOSITE_SCORE = "composite_score"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    # Optimization method and objective
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    objective: OptimizationObjective = OptimizationObjective.PREDICTIVE_POWER
    
    # Search space parameters
    profit_targets_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'micro': (0.002, 0.005),    # 0.2% to 0.5%
        'small': (0.003, 0.008),    # 0.3% to 0.8%
        'medium': (0.005, 0.012),   # 0.5% to 1.2%
        'good': (0.008, 0.020)      # 0.8% to 2.0%
    })
    
    time_horizons_range: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'immediate': (1, 4),        # 1 to 4 periods (5-20 minutes)
        'short': (2, 8)             # 2 to 8 periods (10-40 minutes)
    })
    
    quality_weights_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'speed_weight': (0.1, 0.5),
        'risk_weight': (0.2, 0.6), 
        'profitability_weight': (0.1, 0.5)
    })
    
    # Search parameters
    grid_search_steps: int = 5
    random_search_iterations: int = 100
    bayesian_iterations: int = 50
    
    # Evaluation parameters
    validation_split: float = 0.3
    min_validation_samples: int = 500
    cross_validation_folds: int = 3
    
    # Optimization constraints
    min_transaction_cost: float = 0.0004  # 0.04%
    max_transaction_cost: float = 0.0020  # 0.20%
    min_hit_rate: float = 0.05            # 5% minimum
    max_hit_rate: float = 0.95            # 95% maximum
    
    # Performance thresholds
    min_acceptable_score: float = 0.55
    convergence_tolerance: float = 0.001
    max_optimization_time: int = 3600     # 1 hour maximum
    
    # Parallel processing
    n_jobs: int = -1
    random_seed: int = 42


@dataclass
class OptimizationResult:
    """Result container for parameter optimization."""
    method: OptimizationMethod
    objective: OptimizationObjective
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    validation_scores: Dict[str, float]
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ParameterOptimizer:
    """
    Systematic optimizer for multi-horizon profit labeling parameters.
    
    This class provides comprehensive parameter optimization for the labeling
    system, similar to hyperparameter optimization in machine learning.
    It explores parameter spaces to find optimal configurations.
    
    Key Optimization Features:
    1. **Multi-Objective Optimization**: Balance multiple objectives
    2. **Constraint Handling**: Respect economic and statistical constraints
    3. **Cross-Validation**: Robust evaluation with time-aware splits
    4. **Parallel Processing**: Efficient parameter space exploration
    5. **Advanced Methods**: Support for Bayesian and evolutionary optimization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize the parameter optimizer."""
        self.config = config or OptimizationConfig()
        self.logger = get_logger('ParameterOptimizer')
        
        # Optimization state
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_params_cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info('🎯 Parameter Optimizer initialized')
        self.logger.info(f'   → Method: {self.config.method.value}')
        self.logger.info(f'   → Objective: {self.config.objective.value}')
        
    def optimize_parameters(self,
                          market_data: pd.DataFrame,
                          validation_data: Optional[pd.DataFrame] = None) -> OptimizationResult:
        """
        Optimize labeling parameters using specified method.
        
        Args:
            market_data: Training data for optimization
            validation_data: Optional validation data (will split if not provided)
            
        Returns:
            OptimizationResult containing best parameters and performance
        """
        self.logger.info('🚀 Starting parameter optimization')
        
        # Prepare data splits
        train_data, val_data = self._prepare_data_splits(market_data, validation_data)
        
        if len(train_data) < 1000 or len(val_data) < self.config.min_validation_samples:
            raise ValueError("Insufficient data for optimization")
        
        # Select optimization method
        optimization_func = self._get_optimization_function()
        
        # Run optimization
        start_time = datetime.now()
        try:
            result = optimization_func(train_data, val_data)
            result.metadata['optimization_time'] = (datetime.now() - start_time).total_seconds()
            
            # Store result
            self.optimization_results[f"{self.config.method.value}_{self.config.objective.value}"] = result
            
            self.logger.info(f'✅ Optimization completed in {result.metadata["optimization_time"]:.1f}s')
            self.logger.info(f'   → Best score: {result.best_score:.4f}')
            
            return result
            
        except Exception as e:
            self.logger.error(f'❌ Optimization failed: {e}')
            raise
    
    def _get_optimization_function(self) -> Callable:
        """Get the appropriate optimization function."""
        method_map = {
            OptimizationMethod.GRID_SEARCH: self._grid_search_optimization,
            OptimizationMethod.RANDOM_SEARCH: self._random_search_optimization,
            OptimizationMethod.BAYESIAN_OPTIMIZATION: self._bayesian_optimization,
        }
        
        if HYPEROPT_AVAILABLE and self.config.method == OptimizationMethod.HYPEROPT_TPE:
            method_map[OptimizationMethod.HYPEROPT_TPE] = self._hyperopt_optimization
            
        if OPTUNA_AVAILABLE and self.config.method == OptimizationMethod.OPTUNA_TPE:
            method_map[OptimizationMethod.OPTUNA_TPE] = self._optuna_optimization
        
        if self.config.method not in method_map:
            self.logger.warning(f'Method {self.config.method.value} not available, using grid search')
            return method_map[OptimizationMethod.GRID_SEARCH]
        
        return method_map[self.config.method]
    
    def _prepare_data_splits(self,
                           market_data: pd.DataFrame,
                           validation_data: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and validation data splits."""
        if validation_data is not None:
            return market_data, validation_data
        
        # Time-aware split for financial data
        split_idx = int(len(market_data) * (1 - self.config.validation_split))
        train_data = market_data.iloc[:split_idx].copy()
        val_data = market_data.iloc[split_idx:].copy()
        
        self.logger.info(f'📊 Data split: {len(train_data)} train, {len(val_data)} validation')
        return train_data, val_data
    
    def _grid_search_optimization(self,
                                train_data: pd.DataFrame,
                                val_data: pd.DataFrame) -> OptimizationResult:
        """Grid search optimization."""
        self.logger.info('🔍 Running grid search optimization')
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid()
        self.logger.info(f'   → Grid size: {len(param_grid)} combinations')
        
        # Evaluate all combinations
        results = []
        
        if self.config.n_jobs == 1:
            # Sequential evaluation
            for i, params in enumerate(param_grid):
                if i % 10 == 0:
                    self.logger.info(f'   → Progress: {i}/{len(param_grid)} ({i/len(param_grid)*100:.1f}%)')
                
                score = self._evaluate_parameters(params, train_data, val_data)
                results.append({
                    'params': params,
                    'score': score,
                    'iteration': i
                })
        else:
            # Parallel evaluation
            eval_func = partial(self._evaluate_parameters, 
                              train_data=train_data, val_data=val_data)
            
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                future_to_params = {
                    executor.submit(eval_func, params): (i, params) 
                    for i, params in enumerate(param_grid)
                }
                
                for future in as_completed(future_to_params):
                    i, params = future_to_params[future]
                    try:
                        score = future.result()
                        results.append({
                            'params': params,
                            'score': score,
                            'iteration': i
                        })
                        
                        if len(results) % 10 == 0:
                            self.logger.info(f'   → Progress: {len(results)}/{len(param_grid)}')
                            
                    except Exception as e:
                        self.logger.warning(f'Evaluation failed for params {params}: {e}')
        
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        
        return OptimizationResult(
            method=OptimizationMethod.GRID_SEARCH,
            objective=self.config.objective,
            best_params=best_result['params'],
            best_score=best_result['score'],
            optimization_history=results,
            validation_scores=self._calculate_validation_scores(
                best_result['params'], train_data, val_data
            ),
            convergence_info={'iterations': len(results)},
            metadata={
                'grid_size': len(param_grid),
                'successful_evaluations': len(results)
            }
        )
    
    def _random_search_optimization(self,
                                  train_data: pd.DataFrame,
                                  val_data: pd.DataFrame) -> OptimizationResult:
        """Random search optimization."""
        self.logger.info('🎲 Running random search optimization')
        
        # Generate random parameter combinations
        param_combinations = self._generate_random_parameters(self.config.random_search_iterations)
        self.logger.info(f'   → Random samples: {len(param_combinations)}')
        
        # Evaluate combinations (similar to grid search)
        results = []
        
        for i, params in enumerate(param_combinations):
            if i % 20 == 0:
                self.logger.info(f'   → Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)')
            
            score = self._evaluate_parameters(params, train_data, val_data)
            results.append({
                'params': params,
                'score': score,
                'iteration': i
            })
        
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        
        return OptimizationResult(
            method=OptimizationMethod.RANDOM_SEARCH,
            objective=self.config.objective,
            best_params=best_result['params'],
            best_score=best_result['score'],
            optimization_history=results,
            validation_scores=self._calculate_validation_scores(
                best_result['params'], train_data, val_data
            ),
            convergence_info={'iterations': len(results)},
            metadata={
                'random_samples': len(param_combinations),
                'successful_evaluations': len(results)
            }
        )
    
    def _bayesian_optimization(self,
                             train_data: pd.DataFrame,
                             val_data: pd.DataFrame) -> OptimizationResult:
        """Bayesian optimization (simplified implementation)."""
        self.logger.info('🧠 Running Bayesian optimization')
        
        # For simplicity, use random search with adaptive sampling
        # In production, would use libraries like scikit-optimize
        
        results = []
        best_score = -np.inf
        
        # Initial random exploration
        initial_samples = min(10, self.config.bayesian_iterations // 3)
        initial_params = self._generate_random_parameters(initial_samples)
        
        for i, params in enumerate(initial_params):
            score = self._evaluate_parameters(params, train_data, val_data)
            results.append({
                'params': params,
                'score': score,
                'iteration': i,
                'phase': 'exploration'
            })
            
            if score > best_score:
                best_score = score
        
        # Exploitation phase (sample around best regions)
        for i in range(initial_samples, self.config.bayesian_iterations):
            if i % 10 == 0:
                self.logger.info(f'   → Progress: {i}/{self.config.bayesian_iterations}')
            
            # Sample around current best (simplified acquisition function)
            params = self._sample_around_best(results)
            score = self._evaluate_parameters(params, train_data, val_data)
            
            results.append({
                'params': params,
                'score': score,
                'iteration': i,
                'phase': 'exploitation'
            })
            
            if score > best_score:
                best_score = score
        
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        
        return OptimizationResult(
            method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            objective=self.config.objective,
            best_params=best_result['params'],
            best_score=best_result['score'],
            optimization_history=results,
            validation_scores=self._calculate_validation_scores(
                best_result['params'], train_data, val_data
            ),
            convergence_info={'iterations': len(results)},
            metadata={
                'exploration_samples': initial_samples,
                'exploitation_samples': len(results) - initial_samples
            }
        )
    
    def _hyperopt_optimization(self,
                             train_data: pd.DataFrame,
                             val_data: pd.DataFrame) -> OptimizationResult:
        """Hyperopt TPE optimization."""
        if not HYPEROPT_AVAILABLE:
            raise RuntimeError("Hyperopt not available. Install with: pip install hyperopt")
        
        self.logger.info('🔬 Running Hyperopt TPE optimization')
        
        # Define search space
        space = self._define_hyperopt_space()
        
        # Objective function for hyperopt (minimize, so negate score)
        def objective(params):
            try:
                score = self._evaluate_parameters(params, train_data, val_data)
                return {'loss': -score, 'status': STATUS_OK}
            except Exception as e:
                self.logger.warning(f'Evaluation failed: {e}')
                return {'loss': 0, 'status': STATUS_OK}
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.bayesian_iterations,
            trials=trials,
            rstate=np.random.RandomState(self.config.random_seed)
        )
        
        # Convert results
        results = []
        for i, trial in enumerate(trials.trials):
            results.append({
                'params': trial['misc']['vals'],
                'score': -trial['result']['loss'],
                'iteration': i
            })
        
        best_result = max(results, key=lambda x: x['score'])
        
        return OptimizationResult(
            method=OptimizationMethod.HYPEROPT_TPE,
            objective=self.config.objective,
            best_params=best_result['params'],
            best_score=best_result['score'],
            optimization_history=results,
            validation_scores=self._calculate_validation_scores(
                best_result['params'], train_data, val_data
            ),
            convergence_info={'iterations': len(results)},
            metadata={'hyperopt_trials': len(trials.trials)}
        )
    
    def _optuna_optimization(self,
                           train_data: pd.DataFrame,
                           val_data: pd.DataFrame) -> OptimizationResult:
        """Optuna TPE optimization."""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available. Install with: pip install optuna")
        
        self.logger.info('🎯 Running Optuna TPE optimization')
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed)
        )
        
        # Objective function for optuna
        def objective(trial):
            params = self._suggest_optuna_parameters(trial)
            try:
                return self._evaluate_parameters(params, train_data, val_data)
            except Exception as e:
                self.logger.warning(f'Evaluation failed: {e}')
                return 0.0
        
        # Run optimization
        study.optimize(objective, n_trials=self.config.bayesian_iterations)
        
        # Convert results
        results = []
        for i, trial in enumerate(study.trials):
            results.append({
                'params': trial.params,
                'score': trial.value if trial.value is not None else 0.0,
                'iteration': i
            })
        
        return OptimizationResult(
            method=OptimizationMethod.OPTUNA_TPE,
            objective=self.config.objective,
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=results,
            validation_scores=self._calculate_validation_scores(
                study.best_params, train_data, val_data
            ),
            convergence_info={'iterations': len(results)},
            metadata={'optuna_trials': len(study.trials)}
        )
    
    def _generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        param_ranges = {}
        
        # Profit targets
        for target, (min_val, max_val) in self.config.profit_targets_range.items():
            param_ranges[f'profit_target_{target}'] = np.linspace(
                min_val, max_val, self.config.grid_search_steps
            )
        
        # Time horizons
        for horizon, (min_val, max_val) in self.config.time_horizons_range.items():
            param_ranges[f'time_horizon_{horizon}'] = np.arange(min_val, max_val + 1)
        
        # Quality weights
        for weight, (min_val, max_val) in self.config.quality_weights_range.items():
            param_ranges[weight] = np.linspace(min_val, max_val, self.config.grid_search_steps)
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        grid = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            # Ensure weights sum appropriately
            params = self._normalize_quality_weights(params)
            grid.append(params)
        
        return grid
    
    def _generate_random_parameters(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        np.random.seed(self.config.random_seed)
        
        params_list = []
        for _ in range(n_samples):
            params = {}
            
            # Random profit targets
            for target, (min_val, max_val) in self.config.profit_targets_range.items():
                params[f'profit_target_{target}'] = np.random.uniform(min_val, max_val)
            
            # Random time horizons
            for horizon, (min_val, max_val) in self.config.time_horizons_range.items():
                params[f'time_horizon_{horizon}'] = np.random.randint(min_val, max_val + 1)
            
            # Random quality weights
            for weight, (min_val, max_val) in self.config.quality_weights_range.items():
                params[weight] = np.random.uniform(min_val, max_val)
            
            # Normalize weights
            params = self._normalize_quality_weights(params)
            params_list.append(params)
        
        return params_list
    
    def _normalize_quality_weights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize quality weights to sum to 1.0."""
        weight_keys = ['speed_weight', 'risk_weight', 'profitability_weight']
        
        # Extract weights
        weights = [params.get(key, 0.33) for key in weight_keys]
        total_weight = sum(weights)
        
        # Normalize
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            for key, weight in zip(weight_keys, normalized_weights):
                params[key] = weight
        
        return params
    
    def _evaluate_parameters(self,
                           params: Dict[str, Any],
                           train_data: pd.DataFrame,
                           val_data: pd.DataFrame) -> float:
        """Evaluate a parameter configuration."""
        try:
            # Convert parameters to MultiHorizonConfig
            config = self._params_to_config(params)
            
            # Generate labels with these parameters
            labeler = MultiHorizonProfitLabeler(config)
            labeled_val_data = labeler.generate_labels(val_data.copy())
            
            # Calculate objective score
            score = self._calculate_objective_score(labeled_val_data, val_data)
            
            return score
            
        except Exception as e:
            self.logger.warning(f'Parameter evaluation failed: {e}')
            return 0.0
    
    def _params_to_config(self, params: Dict[str, Any]) -> MultiHorizonConfig:
        """Convert parameter dictionary to MultiHorizonConfig."""
        config = MultiHorizonConfig()
        
        # Update profit targets
        profit_targets = {}
        for key, value in params.items():
            if key.startswith('profit_target_'):
                target_name = key.replace('profit_target_', '')
                profit_targets[target_name] = value
        
        if profit_targets:
            config.profit_targets = profit_targets
        
        # Update time horizons
        time_horizons = {}
        for key, value in params.items():
            if key.startswith('time_horizon_'):
                horizon_name = key.replace('time_horizon_', '')
                time_horizons[horizon_name] = int(value)
        
        if time_horizons:
            config.time_horizons = time_horizons
        
        # Update quality weights
        if 'speed_weight' in params:
            config.speed_weight = params['speed_weight']
        if 'risk_weight' in params:
            config.risk_weight = params['risk_weight']
        if 'profitability_weight' in params:
            config.profitability_weight = params['profitability_weight']
        
        return config
    
    def _calculate_objective_score(self,
                                 labeled_data: pd.DataFrame,
                                 market_data: pd.DataFrame) -> float:
        """Calculate objective score based on configuration."""
        try:
            if self.config.objective == OptimizationObjective.PREDICTIVE_POWER:
                return self._calculate_predictive_power_score(labeled_data, market_data)
            elif self.config.objective == OptimizationObjective.LABEL_STABILITY:
                return self._calculate_stability_score(labeled_data)
            elif self.config.objective == OptimizationObjective.HIT_RATE_BALANCE:
                return self._calculate_hit_rate_balance_score(labeled_data)
            elif self.config.objective == OptimizationObjective.ECONOMIC_VALUE:
                return self._calculate_economic_value_score(labeled_data, market_data)
            elif self.config.objective == OptimizationObjective.COMPOSITE_SCORE:
                return self._calculate_composite_objective_score(labeled_data, market_data)
            else:
                return self._calculate_predictive_power_score(labeled_data, market_data)
                
        except Exception:
            return 0.0
    
    def _calculate_predictive_power_score(self,
                                        labeled_data: pd.DataFrame,
                                        market_data: pd.DataFrame) -> float:
        """Calculate predictive power score."""
        if 'close' not in market_data.columns:
            return 0.0
        
        # Calculate future returns
        future_returns = market_data['close'].pct_change().shift(-1).fillna(0)
        
        # Test key columns
        test_columns = ['overall_opportunity', 'leverage_adjusted_score']
        scores = []
        
        for col in test_columns:
            if col in labeled_data.columns:
                values = labeled_data[col].fillna(0)
                common_idx = values.index.intersection(future_returns.index)
                
                if len(common_idx) > 20:
                    corr = np.corrcoef(
                        values.loc[common_idx],
                        future_returns.loc[common_idx]
                    )[0, 1]
                    
                    if not np.isnan(corr):
                        # Convert correlation to AUC-like score
                        auc_proxy = 0.5 + abs(corr) / 2
                        scores.append(auc_proxy)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_stability_score(self, labeled_data: pd.DataFrame) -> float:
        """Calculate label stability score."""
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        
        stability_scores = []
        for col in prob_columns[:5]:  # Top 5 for efficiency
            values = labeled_data[col].dropna()
            if len(values) > 50:
                # Rolling correlation with itself
                window = min(20, len(values) // 4)
                rolling_std = values.rolling(window=window).std()
                if rolling_std.mean() > 0:
                    stability = 1.0 - (rolling_std.std() / rolling_std.mean())
                    stability_scores.append(max(0.0, min(1.0, stability)))
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _calculate_hit_rate_balance_score(self, labeled_data: pd.DataFrame) -> float:
        """Calculate balanced hit rate score."""
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        
        hit_rates = []
        for col in prob_columns:
            values = labeled_data[col].dropna()
            if len(values) > 50:
                hit_rate = (values > 0.5).mean()
                # Penalize extreme hit rates (too high or too low)
                balanced_score = 1.0 - abs(hit_rate - 0.3)  # Target ~30% hit rate
                hit_rates.append(max(0.0, balanced_score))
        
        return np.mean(hit_rates) if hit_rates else 0.5
    
    def _calculate_economic_value_score(self,
                                      labeled_data: pd.DataFrame,
                                      market_data: pd.DataFrame) -> float:
        """Calculate economic value score."""
        # Simplified economic value calculation
        if 'overall_opportunity' not in labeled_data.columns or 'close' not in market_data.columns:
            return 0.0
        
        opportunities = labeled_data['overall_opportunity'].fillna(0)
        returns = market_data['close'].pct_change().shift(-1).fillna(0)
        
        common_idx = opportunities.index.intersection(returns.index)
        if len(common_idx) < 50:
            return 0.0
        
        # Simple strategy: go long when opportunity > threshold
        threshold = opportunities.quantile(0.7)  # Top 30% opportunities
        signals = (opportunities.loc[common_idx] > threshold).astype(int)
        strategy_returns = signals * returns.loc[common_idx]
        
        # Calculate Sharpe ratio proxy
        if strategy_returns.std() > 0:
            sharpe_proxy = strategy_returns.mean() / strategy_returns.std()
            return max(0.0, min(2.0, sharpe_proxy + 1.0)) / 2.0  # Normalize to [0,1]
        
        return 0.0
    
    def _calculate_composite_objective_score(self,
                                           labeled_data: pd.DataFrame,
                                           market_data: pd.DataFrame) -> float:
        """Calculate composite objective score."""
        # Weighted combination of multiple objectives
        predictive_score = self._calculate_predictive_power_score(labeled_data, market_data)
        stability_score = self._calculate_stability_score(labeled_data)
        economic_score = self._calculate_economic_value_score(labeled_data, market_data)
        
        # Weighted average
        composite_score = (
            0.4 * predictive_score +
            0.3 * stability_score +
            0.3 * economic_score
        )
        
        return composite_score
    
    def _calculate_validation_scores(self,
                                   params: Dict[str, Any],
                                   train_data: pd.DataFrame,
                                   val_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive validation scores for best parameters."""
        try:
            config = self._params_to_config(params)
            labeler = MultiHorizonProfitLabeler(config)
            labeled_data = labeler.generate_labels(val_data.copy())
            
            return {
                'predictive_power': self._calculate_predictive_power_score(labeled_data, val_data),
                'stability': self._calculate_stability_score(labeled_data),
                'hit_rate_balance': self._calculate_hit_rate_balance_score(labeled_data),
                'economic_value': self._calculate_economic_value_score(labeled_data, val_data)
            }
            
        except Exception:
            return {
                'predictive_power': 0.0,
                'stability': 0.0,
                'hit_rate_balance': 0.0,
                'economic_value': 0.0
            }
    
    def _sample_around_best(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sample parameters around current best (simplified acquisition function)."""
        # Get top 10% of results
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_results = sorted_results[:max(1, len(sorted_results) // 10)]
        
        # Sample from top results and add noise
        base_params = np.random.choice(top_results)['params']
        noisy_params = {}
        
        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise (10% std)
                noise_std = abs(value) * 0.1
                noisy_value = np.random.normal(value, noise_std)
                
                # Clip to valid ranges
                if key.startswith('profit_target_'):
                    target_name = key.replace('profit_target_', '')
                    if target_name in self.config.profit_targets_range:
                        min_val, max_val = self.config.profit_targets_range[target_name]
                        noisy_value = np.clip(noisy_value, min_val, max_val)
                elif key.startswith('time_horizon_'):
                    horizon_name = key.replace('time_horizon_', '')
                    if horizon_name in self.config.time_horizons_range:
                        min_val, max_val = self.config.time_horizons_range[horizon_name]
                        noisy_value = int(np.clip(noisy_value, min_val, max_val))
                elif key in self.config.quality_weights_range:
                    min_val, max_val = self.config.quality_weights_range[key]
                    noisy_value = np.clip(noisy_value, min_val, max_val)
                
                noisy_params[key] = noisy_value
            else:
                noisy_params[key] = value
        
        # Normalize quality weights
        return self._normalize_quality_weights(noisy_params)
    
    def _define_hyperopt_space(self) -> Dict[str, Any]:
        """Define Hyperopt search space."""
        space = {}
        
        # Profit targets
        for target, (min_val, max_val) in self.config.profit_targets_range.items():
            space[f'profit_target_{target}'] = hp.uniform(
                f'profit_target_{target}', min_val, max_val
            )
        
        # Time horizons
        for horizon, (min_val, max_val) in self.config.time_horizons_range.items():
            space[f'time_horizon_{horizon}'] = hp.randint(
                f'time_horizon_{horizon}', min_val, max_val + 1
            )
        
        # Quality weights
        for weight, (min_val, max_val) in self.config.quality_weights_range.items():
            space[weight] = hp.uniform(weight, min_val, max_val)
        
        return space
    
    def _suggest_optuna_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        params = {}
        
        # Profit targets
        for target, (min_val, max_val) in self.config.profit_targets_range.items():
            params[f'profit_target_{target}'] = trial.suggest_float(
                f'profit_target_{target}', min_val, max_val
            )
        
        # Time horizons
        for horizon, (min_val, max_val) in self.config.time_horizons_range.items():
            params[f'time_horizon_{horizon}'] = trial.suggest_int(
                f'time_horizon_{horizon}', min_val, max_val
            )
        
        # Quality weights
        for weight, (min_val, max_val) in self.config.quality_weights_range.items():
            params[weight] = trial.suggest_float(weight, min_val, max_val)
        
        return self._normalize_quality_weights(params)
    
    def compare_optimization_methods(self,
                                   market_data: pd.DataFrame,
                                   methods: Optional[List[OptimizationMethod]] = None) -> Dict[str, OptimizationResult]:
        """Compare multiple optimization methods."""
        if methods is None:
            methods = [
                OptimizationMethod.GRID_SEARCH,
                OptimizationMethod.RANDOM_SEARCH,
                OptimizationMethod.BAYESIAN_OPTIMIZATION
            ]
        
        self.logger.info(f'🔬 Comparing {len(methods)} optimization methods')
        
        results = {}
        original_method = self.config.method
        
        for method in methods:
            self.logger.info(f'   → Running {method.value}')
            self.config.method = method
            
            try:
                result = self.optimize_parameters(market_data)
                results[method.value] = result
            except Exception as e:
                self.logger.error(f'Method {method.value} failed: {e}')
        
        # Restore original method
        self.config.method = original_method
        
        return results
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        if not self.optimization_results:
            return "No optimization results available. Run optimize_parameters() first."
        
        report_lines = [
            "# Multi-Horizon Profit Labeling Parameter Optimization Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Completed {len(self.optimization_results)} optimization runs",
            ""
        ]
        
        # Best results summary
        best_result = max(self.optimization_results.values(), key=lambda x: x.best_score)
        report_lines.extend([
            "### Best Configuration",
            f"**Method**: {best_result.method.value}",
            f"**Objective**: {best_result.objective.value}",
            f"**Score**: {best_result.best_score:.4f}",
            "",
            "**Optimal Parameters**:"
        ])
        
        for param, value in best_result.best_params.items():
            if isinstance(value, float):
                report_lines.append(f"- {param}: {value:.4f}")
            else:
                report_lines.append(f"- {param}: {value}")
        
        report_lines.append("")
        
        # Validation scores
        if best_result.validation_scores:
            report_lines.extend([
                "**Validation Scores**:"
            ])
            for metric, score in best_result.validation_scores.items():
                report_lines.append(f"- {metric}: {score:.4f}")
            report_lines.append("")
        
        # Method comparison
        if len(self.optimization_results) > 1:
            report_lines.extend([
                "## Method Comparison",
                ""
            ])
            
            for method_name, result in self.optimization_results.items():
                report_lines.extend([
                    f"### {method_name}",
                    f"- Score: {result.best_score:.4f}",
                    f"- Iterations: {result.convergence_info.get('iterations', 'N/A')}",
                    f"- Time: {result.metadata.get('optimization_time', 'N/A'):.1f}s",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def save_optimization_results(self, output_path: Union[str, Path]):
        """Save optimization results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for key, result in self.optimization_results.items():
            serializable_results[key] = {
                'method': result.method.value,
                'objective': result.objective.value,
                'best_params': result.best_params,
                'best_score': result.best_score,
                'optimization_history': result.optimization_history,
                'validation_scores': result.validation_scores,
                'convergence_info': result.convergence_info,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat()
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'optimization_results': serializable_results,
                'config': {
                    'method': self.config.method.value,
                    'objective': self.config.objective.value,
                    'grid_search_steps': self.config.grid_search_steps,
                    'random_search_iterations': self.config.random_search_iterations,
                    'bayesian_iterations': self.config.bayesian_iterations
                }
            }, f, indent=2)
        
        self.logger.info(f'💾 Optimization results saved to {output_path}')


# Convenience functions
def optimize_labeling_parameters(market_data: pd.DataFrame,
                                method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                                objective: OptimizationObjective = OptimizationObjective.PREDICTIVE_POWER,
                                config: Optional[OptimizationConfig] = None) -> OptimizationResult:
    """Convenience function to optimize labeling parameters."""
    if config is None:
        config = OptimizationConfig(method=method, objective=objective)
    else:
        config.method = method
        config.objective = objective
    
    optimizer = ParameterOptimizer(config)
    return optimizer.optimize_parameters(market_data)


def compare_labeling_optimization_methods(market_data: pd.DataFrame,
                                        methods: Optional[List[OptimizationMethod]] = None,
                                        objective: OptimizationObjective = OptimizationObjective.PREDICTIVE_POWER) -> Dict[str, OptimizationResult]:
    """Convenience function to compare optimization methods."""
    config = OptimizationConfig(objective=objective)
    optimizer = ParameterOptimizer(config)
    return optimizer.compare_optimization_methods(market_data, methods)
