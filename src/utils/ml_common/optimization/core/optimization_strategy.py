"""
Optimization strategy implementations.

This module provides the strategy pattern implementation for different
optimization approaches (Bayesian, Grid, Random, BOHB).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple
import time
from dataclasses import dataclass

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a mock numpy for basic functionality
    class MockNumpy:
        def __getattr__(self, name):
            raise ImportError("NumPy is required for optimization functionality")
    np = MockNumpy()

from ..validation import HPOConfig, SearchSpaceParameter
from ..exceptions import OptimizationError, ModelEvaluationError, TimeoutError
from ..results import HPOResult


@dataclass
class OptimizationContext:
    """Context for optimization execution."""
    model_factory: Callable
    X: Any  # Changed from np.ndarray to Any for compatibility
    y: Any  # Changed from np.ndarray to Any for compatibility
    search_space: Dict[str, SearchSpaceParameter]
    model_name: str
    start_time: float
    config: HPOConfig


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.trial_results = []
        self.optimization_history = []
    
    @abstractmethod
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute optimization strategy."""
        pass
    
    def _check_timeout(self, start_time: float) -> None:
        """Check if optimization has timed out."""
        if self.config.timeout is not None:
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout:
                raise TimeoutError(f"Optimization timed out after {elapsed:.2f}s", {
                    'timeout': self.config.timeout,
                    'elapsed': elapsed
                })
    
    def _evaluate_model(self, model: Any, X: Any, y: Any, 
                       trial_number: int = None) -> float:
        """Evaluate model performance with error handling."""
        try:
            # This would be implemented by the specific strategy
            # or delegated to a shared evaluation service
            return self._perform_evaluation(model, X, y)
        except Exception as e:
            raise ModelEvaluationError(f"Model evaluation failed: {e}", {
                'trial_number': trial_number,
                'model_type': type(model).__name__
            }) from e
    
    def _perform_evaluation(self, model: Any, X: Any, y: Any) -> float:
        """Perform the actual model evaluation."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for model evaluation")
        
        # This is a placeholder - would be implemented with proper CV
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=self.config.cv_folds, 
                                   scoring=self.config.scoring, n_jobs=1)
            return float(np.mean(scores))
        except Exception as e:
            raise ModelEvaluationError(f"Cross-validation failed: {e}") from e


class BayesianStrategy(OptimizationStrategy):
    """Bayesian optimization using TPE."""
    
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute Bayesian optimization."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise OptimizationError("Optuna is required for Bayesian optimization")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                seed=self.config.random_state
            )
        )
        
        # Define objective
        def objective(trial):
            self._check_timeout(context.start_time)
            params = self._sample_parameters_from_trial(trial, context.search_space)
            model = context.model_factory(**params)
            return self._evaluate_model(model, context.X, context.y, trial.number)
        
        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, 
                      timeout=self.config.timeout)
        
        # Extract results
        best_trial = study.best_trial
        trial_results = [trial for trial in study.trials]
        
        return HPOResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            best_trial=best_trial,
            n_trials=len(trial_results),
            trial_results=[{
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            } for trial in trial_results],
            mean_score=np.mean([t.value for t in trial_results if t.value is not None]),
            std_score=np.std([t.value for t in trial_results if t.value is not None]),
            min_score=np.min([t.value for t in trial_results if t.value is not None]),
            max_score=np.max([t.value for t in trial_results if t.value is not None]),
            strategy=self.config.strategy.value,
            optimization_time=time.time() - context.start_time
        )
    
    def _sample_parameters_from_trial(self, trial, search_space: Dict[str, SearchSpaceParameter]) -> Dict[str, Any]:
        """Sample parameters from Optuna trial."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config.type.value == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config.low, param_config.high, 
                    log=param_config.log
                )
            elif param_config.type.value == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config.low, param_config.high, 
                    log=param_config.log
                )
            elif param_config.type.value == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config.choices)
            else:
                raise OptimizationError(f"Unsupported parameter type: {param_config.type}")
        
        return params


class GridStrategy(OptimizationStrategy):
    """Grid search optimization with staged refinement."""
    
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute grid search optimization."""
        # Stage 1: Coarse grid search
        coarse_grid = self._generate_parameter_grid(context.search_space, self.config.coarse_grid_points)
        stage1_results = self._eval_param_list(context, coarse_grid)
        
        # Pick top K from coarse stage
        top = sorted(stage1_results, key=lambda d: d['value'], reverse=True)[:self.config.coarse_grid_trials]
        best_score = top[0]['value'] if top else -np.inf
        best_params = top[0]['params'] if top else {}
        
        # Stage 2: Fine grid search (optional)
        trial_results = stage1_results[:]
        if self.config.enable_staged_optimization and top:
            refined_space = self._refine_search_space([t['params'] for t in top], context.search_space)
            fine_grid = self._generate_parameter_grid(refined_space, self.config.fine_grid_points)
            stage2_results = self._eval_param_list(context, fine_grid)
            trial_results.extend(stage2_results)
            
            # Update best if we found something better
            for r in stage2_results:
                if r['value'] > best_score:
                    best_score, best_params = r['value'], r['params']
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trial_results),
            trial_results=trial_results,
            mean_score=float(np.mean([t['value'] for t in trial_results])) if trial_results else -np.inf,
            std_score=float(np.std([t['value'] for t in trial_results])) if trial_results else 0.0,
            min_score=float(np.min([t['value'] for t in trial_results])) if trial_results else -np.inf,
            max_score=float(np.max([t['value'] for t in trial_results])) if trial_results else -np.inf,
            strategy=self.config.strategy.value,
            optimization_time=time.time() - context.start_time
        )
    
    def _generate_parameter_grid(self, search_space: Dict[str, SearchSpaceParameter], points: int) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        import itertools
        
        param_combinations = []
        
        for param_name, param_config in search_space.items():
            if param_config.type.value == 'float':
                if param_config.log:
                    values = np.logspace(
                        np.log10(param_config.low), 
                        np.log10(param_config.high), 
                        points
                    )
                else:
                    values = np.linspace(
                        param_config.low, 
                        param_config.high, 
                        points
                    )
                param_combinations.append([(param_name, v) for v in values])
            elif param_config.type.value == 'int':
                values = np.unique(np.linspace(
                    param_config.low, 
                    param_config.high, 
                    points, 
                    dtype=int
                ))
                param_combinations.append([(param_name, v) for v in values])
            elif param_config.type.value == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config.choices])
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))
        
        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)
        
        return grid
    
    def _eval_param_list(self, context: OptimizationContext, param_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a list of parameter combinations."""
        out = []
        for i, params in enumerate(param_list):
            try:
                model = context.model_factory(**params)
                score = self._evaluate_model(model, context.X, context.y, i)
                out.append({'trial_number': i, 'params': params, 'value': score, 'state': 'COMPLETE'})
            except Exception as e:
                # Log error but continue
                continue
        return out
    
    def _refine_search_space(self, winners: List[Dict[str, Any]], base_space: Dict[str, SearchSpaceParameter]) -> Dict[str, SearchSpaceParameter]:
        """Refine search space around winning configurations."""
        refined = {}
        for name, cfg in base_space.items():
            if cfg.type.value == 'categorical':
                # Keep observed best categories
                seen = list({w[name] for w in winners if name in w})
                refined[name] = SearchSpaceParameter(
                    type=cfg.type,
                    choices=seen or cfg.choices
                )
            elif cfg.type.value in ('float', 'int'):
                vals = np.array([w[name] for w in winners if name in w])
                if len(vals) > 0:
                    lo, hi = np.min(vals), np.max(vals)
                    span = (hi - lo)
                    if span == 0:
                        lo, hi = cfg.low, cfg.high  # fallback to base range
                    pad = max(span * 0.3, (cfg.high - cfg.low) * 0.05)
                    lo2 = max(cfg.low, lo - pad)
                    hi2 = min(cfg.high, hi + pad)
                    refined[name] = SearchSpaceParameter(
                        type=cfg.type,
                        low=float(lo2),
                        high=float(hi2),
                        log=cfg.log
                    )
                else:
                    refined[name] = cfg
            else:
                raise OptimizationError(f"Unsupported type {cfg.type}")
        return refined


class RandomStrategy(OptimizationStrategy):
    """Random search optimization."""
    
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute random search optimization."""
        best_score = -np.inf
        best_params = {}
        trial_results = []
        
        for i in range(self.config.n_trials):
            try:
                self._check_timeout(context.start_time)
                
                # Sample random parameters
                params = self._sample_parameters(context.search_space)
                
                # Create model with parameters
                model = context.model_factory(**params)
                
                # Evaluate model
                score = self._evaluate_model(model, context.X, context.y, i)
                
                trial_results.append({
                    'trial_number': i,
                    'params': params,
                    'value': score,
                    'state': 'COMPLETE'
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
            except Exception as e:
                # Log error but continue
                continue
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trial_results),
            trial_results=trial_results,
            mean_score=np.mean([t['value'] for t in trial_results]),
            std_score=np.std([t['value'] for t in trial_results]),
            min_score=np.min([t['value'] for t in trial_results]),
            max_score=np.max([t['value'] for t in trial_results]),
            strategy=self.config.strategy.value,
            optimization_time=time.time() - context.start_time
        )
    
    def _sample_parameters(self, search_space: Dict[str, SearchSpaceParameter]) -> Dict[str, Any]:
        """Sample parameters randomly."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config.type.value == 'float':
                if param_config.log:
                    params[param_name] = np.exp(np.random.uniform(
                        np.log(param_config.low), np.log(param_config.high)
                    ))
                else:
                    params[param_name] = np.random.uniform(
                        param_config.low, param_config.high
                    )
            elif param_config.type.value == 'int':
                params[param_name] = np.random.randint(
                    param_config.low, param_config.high + 1
                )
            elif param_config.type.value == 'categorical':
                params[param_name] = np.random.choice(param_config.choices)
            else:
                raise OptimizationError(f"Unsupported parameter type: {param_config.type}")
        
        return params


class BOHBStrategy(OptimizationStrategy):
    """BOHB-style multi-fidelity optimization."""
    
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute BOHB optimization."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise OptimizationError("Optuna is required for BOHB optimization")
        
        # Create study with pruner
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                seed=self.config.random_state
            )
        )
        
        # Define objective with multi-fidelity support
        def objective(trial):
            self._check_timeout(context.start_time)
            params = self._sample_parameters_from_trial(trial, context.search_space)
            
            # Sample budget (fidelity level)
            budget = trial.suggest_float('budget', self.config.min_budget, self.config.max_budget)
            
            # Create model with sampled parameters
            model = context.model_factory(**params)
            
            # Evaluate model with limited budget
            score = self._evaluate_model_with_budget(model, context.X, context.y, budget)
            
            # Report intermediate result for pruning
            trial.report(score, step=int(budget * 100))
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
        
        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, 
                      timeout=self.config.timeout)
        
        # Extract results
        best_trial = study.best_trial
        trial_results = [trial for trial in study.trials]
        
        return HPOResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            best_trial=best_trial,
            n_trials=len(trial_results),
            trial_results=[{
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            } for trial in trial_results],
            mean_score=np.mean([t.value for t in trial_results if t.value is not None]),
            std_score=np.std([t.value for t in trial_results if t.value is not None]),
            min_score=np.min([t.value for t in trial_results if t.value is not None]),
            max_score=np.max([t.value for t in trial_results if t.value is not None]),
            strategy=self.config.strategy.value,
            optimization_time=time.time() - context.start_time
        )
    
    def _sample_parameters_from_trial(self, trial, search_space: Dict[str, SearchSpaceParameter]) -> Dict[str, Any]:
        """Sample parameters from Optuna trial."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config.type.value == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config.low, param_config.high, 
                    log=param_config.log
                )
            elif param_config.type.value == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config.low, param_config.high, 
                    log=param_config.log
                )
            elif param_config.type.value == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config.choices)
            else:
                raise OptimizationError(f"Unsupported parameter type: {param_config.type}")
        
        return params
    
    def _evaluate_model_with_budget(self, model: Any, X: np.ndarray, y: np.ndarray, budget: float) -> float:
        """Evaluate model with limited budget (for multi-fidelity)."""
        try:
            # Use a subset of data based on budget
            budget = max(0.01, min(1.0, budget))  # Clamp to [0.01, 1.0]
            n_samples = int(len(X) * budget)
            n_samples = max(10, min(n_samples, len(X)))  # Ensure valid range
            
            # Sample data with replacement if needed
            replace = n_samples > len(X)
            indices = np.random.choice(len(X), n_samples, replace=replace)
            X_subset = X[indices]
            y_subset = y[indices]
            
            # Evaluate model
            return self._evaluate_model(model, X_subset, y_subset)
            
        except Exception as e:
            raise ModelEvaluationError(f"Multi-fidelity model evaluation failed: {e}") from e