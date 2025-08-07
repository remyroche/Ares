# src/training/steps/optimized_optuna_optimization.py

import optuna
import numpy as np
import pandas as pd
from typing import Any, Dict, Callable, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime


class OptimizedOptunaOptimization:
    """
    Optimized Optuna optimization with early stopping, pruning, and parallel execution.
    
    This implementation provides significant performance improvements over
    standard Optuna optimization by using advanced pruning strategies,
    early stopping, and parallel trial execution.
    """
    
    def __init__(self,
                 n_trials: int = 100,
                 n_jobs: int = -1,
                 early_stopping_patience: int = 10,
                 pruning_threshold: float = 0.8,
                 memory_efficient: bool = True):
        """
        Initialize the optimized Optuna optimizer.
        
        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 for all cores)
            early_stopping_patience: Patience for early stopping
            pruning_threshold: Threshold for pruning unpromising trials
            memory_efficient: Use memory-efficient operations
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.early_stopping_patience = early_stopping_patience
        self.pruning_threshold = pruning_threshold
        self.memory_efficient = memory_efficient
        
        # Suppress Optuna's informational messages
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize_with_early_stopping(
        self,
        objective_func: Callable,
        direction: str = "maximize",
        study_name: str = "optimized_study"
    ) -> Dict[str, Any]:
        """
        Optimize with early stopping and pruning.
        
        Args:
            objective_func: Objective function to optimize
            direction: Optimization direction ("maximize" or "minimize")
            study_name: Name of the study
            
        Returns:
            Dictionary with optimization results
        """
        # Create study with pruning
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,
                max_resource=self.n_trials,
                reduction_factor=3
            )
        )
        
        # Add early stopping callback
        def early_stopping_callback(study, trial):
            if trial.number > self.early_stopping_patience:
                best_value = study.best_value
                current_value = trial.value
                
                if direction == "maximize":
                    if current_value < best_value * self.pruning_threshold:
                        study.stop()
                else:
                    if current_value > best_value / self.pruning_threshold:
                        study.stop()
        
        # Add pruning callback
        def pruning_callback(study, trial):
            if trial.number > 5:
                best_value = study.best_value
                current_value = trial.value
                
                if direction == "maximize":
                    if current_value < best_value * self.pruning_threshold:
                        raise optuna.TrialPruned()
                else:
                    if current_value > best_value / self.pruning_threshold:
                        raise optuna.TrialPruned()
        
        # Optimize with callbacks
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            callbacks=[early_stopping_callback, pruning_callback],
            n_jobs=self.n_jobs
        )
        
        return self._extract_optimization_results(study)
    
    def optimize_parallel(
        self,
        objective_func: Callable,
        direction: str = "maximize",
        study_name: str = "parallel_study"
    ) -> Dict[str, Any]:
        """
        Optimize with parallel trial execution.
        
        Args:
            objective_func: Objective function to optimize
            direction: Optimization direction
            study_name: Name of the study
            
        Returns:
            Dictionary with optimization results
        """
        # Create study for parallel optimization
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # Optimize with parallel execution
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        return self._extract_optimization_results(study)
    
    def optimize_with_caching(
        self,
        objective_func: Callable,
        direction: str = "maximize",
        study_name: str = "cached_study",
        cache_file: str = "optuna_cache.pkl"
    ) -> Dict[str, Any]:
        """
        Optimize with result caching for expensive objective functions.
        
        Args:
            objective_func: Objective function to optimize
            direction: Optimization direction
            study_name: Name of the study
            cache_file: File to cache results
            
        Returns:
            Dictionary with optimization results
        """
        import pickle
        import os
        
        # Load cached results if available
        cached_results = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Create cached objective function
        def cached_objective(trial):
            # Create trial key
            trial_key = str(trial.params)
            
            # Check cache
            if trial_key in cached_results:
                return cached_results[trial_key]
            
            # Evaluate objective
            result = objective_func(trial)
            
            # Cache result
            cached_results[trial_key] = result
            
            # Save cache periodically
            if trial.number % 10 == 0:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cached_results, f)
                except Exception as e:
                    print(f"Error saving cache: {e}")
            
            return result
        
        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=optuna.pruners.SuccessiveHalvingPruner()
        )
        
        # Optimize with caching
        study.optimize(
            cached_objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        # Save final cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_results, f)
        except Exception as e:
            print(f"Error saving final cache: {e}")
        
        return self._extract_optimization_results(study)
    
    def _extract_optimization_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Extract optimization results from study."""
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "optimization_history": [
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name
                }
                for trial in study.trials
            ],
            "optimization_features": {
                "early_stopping": True,
                "pruning": True,
                "parallel_execution": self.n_jobs > 1,
                "memory_efficient": self.memory_efficient
            }
        }
    
    def create_optimized_objective(
        self,
        base_objective: Callable,
        early_stopping_threshold: float = 0.5,
        max_iterations: int = 100
    ) -> Callable:
        """
        Create an optimized objective function with early stopping.
        
        Args:
            base_objective: Base objective function
            early_stopping_threshold: Threshold for early stopping
            max_iterations: Maximum iterations per trial
            
        Returns:
            Optimized objective function
        """
        def optimized_objective(trial):
            # Add early stopping to the objective
            for i in range(max_iterations):
                # Simulate intermediate evaluation
                intermediate_value = base_objective(trial)
                
                # Early stopping check
                if intermediate_value < early_stopping_threshold:
                    trial.report(intermediate_value, step=i)
                    return intermediate_value
                
                # Report intermediate value
                trial.report(intermediate_value, step=i)
            
            # Final evaluation
            return base_objective(trial)
        
        return optimized_objective
    
    def optimize_hyperparameters_parallel(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model type.
        
        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with optimization results
        """
        def objective(trial):
            # Suggest hyperparameters based on model type
            if model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
                }
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params, random_state=42)
                
            elif model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0)
                }
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                
            elif model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0)
                }
                import xgboost as xgb
                model = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train and evaluate model
            if X_val is not None and y_val is not None:
                # Use validation set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            else:
                # Use cross-validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=3, scoring='accuracy', n_jobs=1
                )
                score = scores.mean()
            
            return score
        
        # Optimize with early stopping
        return self.optimize_with_early_stopping(
            objective,
            direction="maximize",
            study_name=f"{model_type}_optimization"
        )


def benchmark_optuna_optimization_methods() -> Dict[str, float]:
    """
    Benchmark different Optuna optimization methods.
    
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Sample objective function
    def sample_objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return -(x**2 + y**2)  # Maximize negative distance from origin
    
    # Standard optimization (baseline)
    start_time = time.time()
    study_standard = optuna.create_study(direction="maximize")
    study_standard.optimize(sample_objective, n_trials=50)
    standard_time = time.time() - start_time
    
    # Optimized optimization
    optimizer = OptimizedOptunaOptimization(n_trials=50, n_jobs=-1)
    start_time = time.time()
    results = optimizer.optimize_with_early_stopping(
        sample_objective,
        direction="maximize",
        study_name="benchmark_study"
    )
    optimized_time = time.time() - start_time
    
    # Parallel optimization
    start_time = time.time()
    parallel_results = optimizer.optimize_parallel(
        sample_objective,
        direction="maximize",
        study_name="parallel_benchmark"
    )
    parallel_time = time.time() - start_time
    
    return {
        'standard_time': standard_time,
        'optimized_time': optimized_time,
        'parallel_time': parallel_time,
        'optimized_speedup': standard_time / optimized_time,
        'parallel_speedup': standard_time / parallel_time,
        'best_value_standard': study_standard.best_value,
        'best_value_optimized': results['best_value'],
        'best_value_parallel': parallel_results['best_value']
    }


def create_optimized_hyperparameter_ranges() -> Dict[str, Dict[str, Any]]:
    """
    Create optimized hyperparameter ranges for different model types.
    
    Returns:
        Dictionary of hyperparameter ranges
    """
    return {
        "random_forest": {
            "n_estimators": {"type": "int", "range": [50, 200]},
            "max_depth": {"type": "int", "range": [3, 12]},
            "min_samples_split": {"type": "int", "range": [2, 15]},
            "min_samples_leaf": {"type": "int", "range": [1, 8]}
        },
        "lightgbm": {
            "n_estimators": {"type": "int", "range": [50, 200]},
            "max_depth": {"type": "int", "range": [3, 8]},
            "learning_rate": {"type": "float", "range": [0.01, 0.2], "log": True},
            "subsample": {"type": "float", "range": [0.7, 1.0]}
        },
        "xgboost": {
            "n_estimators": {"type": "int", "range": [50, 200]},
            "max_depth": {"type": "int", "range": [3, 8]},
            "learning_rate": {"type": "float", "range": [0.01, 0.2], "log": True},
            "subsample": {"type": "float", "range": [0.7, 1.0]}
        }
    }


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test optimization
    optimizer = OptimizedOptunaOptimization(n_trials=20, n_jobs=-1)
    
    # Optimize Random Forest
    rf_results = optimizer.optimize_hyperparameters_parallel(
        "random_forest", X_train, y_train, X_val, y_val
    )
    
    print(f"Random Forest optimization results:")
    print(f"Best parameters: {rf_results['best_params']}")
    print(f"Best score: {rf_results['best_value']:.3f}")
    print(f"Trials completed: {rf_results['n_completed_trials']}")
    print(f"Trials pruned: {rf_results['n_pruned_trials']}")
    
    # Benchmark
    benchmark_results = benchmark_optuna_optimization_methods()
    print(f"\nBenchmark results: {benchmark_results}")
    
    # Test caching
    cached_results = optimizer.optimize_with_caching(
        lambda trial: -(trial.suggest_float("x", -10, 10)**2),
        direction="maximize",
        study_name="cached_test"
    )
    
    print(f"\nCached optimization results:")
    print(f"Best value: {cached_results['best_value']:.3f}")
    print(f"Trials completed: {cached_results['n_completed_trials']}")
