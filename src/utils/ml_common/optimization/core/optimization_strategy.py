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

# Try to import tprint
try:
    from ...tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer, tprint_data_preview,
        tprint_data_format, LogLevel, TPrintConfig
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def tprint_success(*args, **kwargs): pass


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
    # Hardware optimization attributes
    max_workers: Optional[int] = None
    memory_limit: Optional[float] = None
    batch_size: Optional[int] = None


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, config: HPOConfig, early_stopping_integration=None):
        self.config = config
        self.trial_results = []
        self.optimization_history = []
        self.early_stopping_integration = early_stopping_integration
        self.trial_history = []
        self.early_stopped = False
        self.early_stopping_reason = None
    
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
            score = self._perform_evaluation(model, X, y)
            
            # Update trial history for early stopping
            if self.early_stopping_integration:
                self.trial_history.append(score)
                
                # Check for early stopping
                if self.early_stopping_integration.should_stop_early(
                    self.config.strategy.value, 
                    self.trial_history, 
                    trial_number or len(self.trial_history),
                    self.config.n_trials
                ):
                    self.early_stopped = True
                    self.early_stopping_reason = self.early_stopping_integration._get_stopping_reason(
                        self.config.strategy.value
                    )
                    if TPRINT_AVAILABLE:
                        tprint_info(f"⏹️ Early stopping triggered at trial {trial_number}")
            
            return score
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
    """Bayesian optimization using TPE with Grid pre-step."""
    
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute Bayesian optimization with Grid pre-step."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise OptimizationError("Optuna is required for Bayesian optimization")
        
        all_trial_results = []
        total_trials = 0
        
        # Step 1: Coarse Grid Search (if enabled)
        if self.config.enable_staged_optimization:
            grid_result = self._run_grid_prestep(context)
            all_trial_results.extend(grid_result.trial_results)
            total_trials += grid_result.n_trials
            
            # Use best grid results to inform Bayesian search
            if grid_result.trial_results:
                best_grid_params = grid_result.best_params
                # Refine search space around best grid results
                context.search_space = self._refine_search_space_around_best(
                    context.search_space, best_grid_params
                )
        
        # Step 2: Bayesian Optimization
        # Calculate remaining trials for Bayesian phase
        remaining_trials = max(1, self.config.n_trials - total_trials)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=min(self.config.n_startup_trials, remaining_trials // 2),
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                seed=self.config.random_state
            )
        )
        
        # Define objective
        def objective(trial):
            self._check_timeout(context.start_time)
            
            # Check for early stopping
            if self.early_stopped:
                raise optuna.TrialPruned()
            
            params = self._sample_parameters_from_trial(trial, context.search_space)
            model = context.model_factory(**params)
            return self._evaluate_model(model, context.X, context.y, trial.number)
        
        # Optimize
        study.optimize(objective, n_trials=remaining_trials, 
                      timeout=self.config.timeout)
        
        # Combine results
        bayesian_trial_results = [{
            'trial_number': trial.number + total_trials,
            'params': trial.params,
            'value': trial.value,
            'state': trial.state.name
        } for trial in study.trials]
        
        all_trial_results.extend(bayesian_trial_results)
        
        # Find overall best result
        best_trial = study.best_trial
        if all_trial_results:
            # Check if grid search found better results
            grid_best = max(all_trial_results[:total_trials], 
                          key=lambda x: x.get('value', -float('inf'))) if total_trials > 0 else None
            bayesian_best = max(bayesian_trial_results, 
                              key=lambda x: x.get('value', -float('inf'))) if bayesian_trial_results else None
            
            if grid_best and bayesian_best:
                if grid_best['value'] > bayesian_best['value']:
                    best_params = grid_best['params']
                    best_score = grid_best['value']
                else:
                    best_params = bayesian_best['params']
                    best_score = bayesian_best['value']
            elif grid_best:
                best_params = grid_best['params']
                best_score = grid_best['value']
            else:
                best_params = bayesian_best['params']
                best_score = bayesian_best['value']
        else:
            best_params = best_trial.params
            best_score = best_trial.value
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            best_trial=best_trial,
            n_trials=len(all_trial_results),
            trial_results=all_trial_results,
            mean_score=np.mean([t['value'] for t in all_trial_results if t.get('value') is not None]),
            std_score=np.std([t['value'] for t in all_trial_results if t.get('value') is not None]),
            min_score=np.min([t['value'] for t in all_trial_results if t.get('value') is not None]),
            max_score=np.max([t['value'] for t in all_trial_results if t.get('value') is not None]),
            strategy=self.config.strategy.value,
            optimization_time=time.time() - context.start_time
        )
    
    def _run_grid_prestep(self, context: OptimizationContext) -> HPOResult:
        """Run coarse grid search as pre-step."""
        # Create a temporary grid strategy
        grid_strategy = GridStrategy(self.config)
        
        # Use coarse grid settings
        original_coarse_points = self.config.coarse_grid_points
        original_coarse_trials = self.config.coarse_grid_trials
        
        # Temporarily modify config for coarse grid
        self.config.coarse_grid_points = max(2, self.config.coarse_grid_points)
        self.config.coarse_grid_trials = min(self.config.coarse_grid_trials, 
                                           self.config.n_trials // 3)
        
        try:
            result = grid_strategy.optimize(context)
            return result
        finally:
            # Restore original settings
            self.config.coarse_grid_points = original_coarse_points
            self.config.coarse_grid_trials = original_coarse_trials
    
    def _refine_search_space_around_best(self, search_space: Dict[str, SearchSpaceParameter], 
                                       best_params: Dict[str, Any]) -> Dict[str, SearchSpaceParameter]:
        """Refine search space around best parameters from grid search."""
        refined_space = {}
        
        for param_name, param_config in search_space.items():
            if param_name in best_params:
                best_value = best_params[param_name]
                
                if param_config.type.value == 'float':
                    # Create tighter range around best value
                    range_size = (param_config.high - param_config.low) * 0.3  # 30% of original range
                    new_low = max(param_config.low, best_value - range_size)
                    new_high = min(param_config.high, best_value + range_size)
                    
                    refined_space[param_name] = SearchSpaceParameter(
                        type=param_config.type,
                        low=new_low,
                        high=new_high,
                        log=param_config.log
                    )
                elif param_config.type.value == 'int':
                    # Create tighter range around best value
                    range_size = max(1, int((param_config.high - param_config.low) * 0.3))
                    new_low = max(param_config.low, best_value - range_size)
                    new_high = min(param_config.high, best_value + range_size)
                    
                    refined_space[param_name] = SearchSpaceParameter(
                        type=param_config.type,
                        low=new_low,
                        high=new_high,
                        log=param_config.log
                    )
                else:
                    # Keep categorical parameters as-is
                    refined_space[param_name] = param_config
            else:
                # Keep parameters not in best_params as-is
                refined_space[param_name] = param_config
        
        return refined_space
    
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
    """Grid search optimization with proper coarse-to-fine progression."""
    
    def optimize(self, context: OptimizationContext) -> HPOResult:
        """Execute grid search optimization with coarse-to-fine progression."""
        all_trial_results = []
        
        # Stage 1: Coarse Grid Search
        if TPRINT_AVAILABLE:
            tprint_info(f"🔍 Stage 1: Coarse grid search with {self.config.coarse_grid_points} points per parameter")
        
        coarse_grid = self._generate_parameter_grid(context.search_space, self.config.coarse_grid_points)
        stage1_results = self._eval_param_list(context, coarse_grid, stage_name="coarse")
        all_trial_results.extend(stage1_results)
        
        # Find top performers from coarse stage
        valid_results = [r for r in stage1_results if r.get('value') is not None]
        if not valid_results:
            raise OptimizationError("No valid results from coarse grid search")
        
        # Sort by score and take top performers
        top_coarse = sorted(valid_results, key=lambda d: d['value'], reverse=True)[:self.config.coarse_grid_trials]
        best_score = top_coarse[0]['value'] if top_coarse else -np.inf
        best_params = top_coarse[0]['params'] if top_coarse else {}
        
        if TPRINT_AVAILABLE:
            tprint_info(f"📊 Coarse stage: {len(valid_results)} valid trials, best score: {best_score:.4f}")
        
        # Stage 2: Fine Grid Search (if enabled and we have good coarse results)
        if self.config.enable_staged_optimization and top_coarse:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔍 Stage 2: Fine grid search with {self.config.fine_grid_points} points per parameter")
            
            # Refine search space around top coarse results
            refined_space = self._refine_search_space_around_winners(
                [t['params'] for t in top_coarse], context.search_space
            )
            
            # Generate fine grid
            fine_grid = self._generate_parameter_grid(refined_space, self.config.fine_grid_points)
            stage2_results = self._eval_param_list(context, fine_grid, stage_name="fine")
            all_trial_results.extend(stage2_results)
            
            # Update best if we found something better in fine stage
            valid_fine = [r for r in stage2_results if r.get('value') is not None]
            if valid_fine:
                best_fine = max(valid_fine, key=lambda d: d['value'])
                if best_fine['value'] > best_score:
                    best_score = best_fine['value']
                    best_params = best_fine['params']
                    
                if TPRINT_AVAILABLE:
                    tprint_info(f"📊 Fine stage: {len(valid_fine)} valid trials, best score: {best_fine['value']:.4f}")
        
        # Stage 3: TPE Refinement (if enabled and we have remaining trials)
        remaining_trials = self.config.n_trials - len(all_trial_results)
        if (self.config.enable_staged_optimization and 
            remaining_trials > 0 and 
            self.config.tpe_trials > 0):
            
            if TPRINT_AVAILABLE:
                tprint_info(f"🔍 Stage 3: TPE refinement with {min(remaining_trials, self.config.tpe_trials)} trials")
            
            tpe_results = self._run_tpe_refinement(context, best_params, min(remaining_trials, self.config.tpe_trials))
            all_trial_results.extend(tpe_results)
            
            # Update best if TPE found something better
            valid_tpe = [r for r in tpe_results if r.get('value') is not None]
            if valid_tpe:
                best_tpe = max(valid_tpe, key=lambda d: d['value'])
                if best_tpe['value'] > best_score:
                    best_score = best_tpe['value']
                    best_params = best_tpe['params']
                    
                if TPRINT_AVAILABLE:
                    tprint_info(f"📊 TPE stage: {len(valid_tpe)} valid trials, best score: {best_tpe['value']:.4f}")
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ Grid optimization completed: {len(all_trial_results)} total trials, best score: {best_score:.4f}")
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(all_trial_results),
            trial_results=all_trial_results,
            mean_score=float(np.mean([t['value'] for t in all_trial_results if t.get('value') is not None])) if all_trial_results else -np.inf,
            std_score=float(np.std([t['value'] for t in all_trial_results if t.get('value') is not None])) if all_trial_results else 0.0,
            min_score=float(np.min([t['value'] for t in all_trial_results if t.get('value') is not None])) if all_trial_results else -np.inf,
            max_score=float(np.max([t['value'] for t in all_trial_results if t.get('value') is not None])) if all_trial_results else -np.inf,
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
    
    def _eval_param_list(self, context: OptimizationContext, param_list: List[Dict[str, Any]], 
                        stage_name: str = "grid") -> List[Dict[str, Any]]:
        """Evaluate a list of parameter combinations."""
        out = []
        for i, params in enumerate(param_list):
            try:
                self._check_timeout(context.start_time)
                
                # Check for early stopping
                if self.early_stopped:
                    if TPRINT_AVAILABLE:
                        tprint_info(f"⏹️ Early stopping triggered in {stage_name} stage")
                    break
                
                model = context.model_factory(**params)
                score = self._evaluate_model(model, context.X, context.y, i)
                out.append({
                    'trial_number': i, 
                    'params': params, 
                    'value': score, 
                    'state': 'COMPLETE',
                    'stage': stage_name
                })
            except Exception as e:
                # Log error but continue
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Trial {i} failed in {stage_name} stage: {e}")
                continue
        return out
    
    def _refine_search_space_around_winners(self, winners: List[Dict[str, Any]], 
                                          base_space: Dict[str, SearchSpaceParameter]) -> Dict[str, SearchSpaceParameter]:
        """Refine search space around winning configurations."""
        if not winners:
            return base_space
            
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
    
    def _run_tpe_refinement(self, context: OptimizationContext, best_params: Dict[str, Any], 
                           n_trials: int) -> List[Dict[str, Any]]:
        """Run TPE refinement around best parameters."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Optuna not available, skipping TPE refinement")
            return []
        
        # Create refined search space around best parameters
        refined_space = self._refine_search_space_around_winners([best_params], context.search_space)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=max(1, n_trials // 4),
                n_ei_candidates=min(24, n_trials),
                multivariate=True,
                group=True,
                seed=self.config.random_state
            )
        )
        
        # Define objective
        def objective(trial):
            self._check_timeout(context.start_time)
            params = self._sample_parameters_from_trial(trial, refined_space)
            model = context.model_factory(**params)
            return self._evaluate_model(model, context.X, context.y, trial.number)
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Convert results
        return [{
            'trial_number': trial.number,
            'params': trial.params,
            'value': trial.value,
            'state': trial.state.name,
            'stage': 'tpe'
        } for trial in study.trials]
    
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
                
                # Check for early stopping
                if self.early_stopped:
                    if TPRINT_AVAILABLE:
                        tprint_info(f"⏹️ Early stopping triggered at trial {i}")
                    break
                
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
            
            # Check for early stopping
            if self.early_stopped:
                raise optuna.TrialPruned()
            
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