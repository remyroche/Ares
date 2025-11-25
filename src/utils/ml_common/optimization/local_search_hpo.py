"""
Adaptive Local Search HPO for XGBoost Models.

Implements efficient hyperparameter optimization using an adaptive grid that:
1. Performs small local searches for each training run
2. Every 6 runs, performs a wider global search
3. Uses early stopping to save computation time
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import logging
from pathlib import Path
import json

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""

    # Local search settings
    local_n_trials: int = 10
    local_search_radius: float = 0.2  # 20% variation from current best

    # Global search settings
    global_n_trials: int = 30
    global_every_n_runs: int = 6

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 50

    # Best parameters tracking
    best_params: Optional[Dict[str, Any]] = None
    run_counter: int = 0


class AdaptiveGrid:
    """
    Adaptive grid for hyperparameter optimization.

    Maintains best parameters and adaptively explores around them,
    with periodic global searches to escape local optima.
    """

    def __init__(
        self,
        config: HPOConfig,
        cache_dir: Path = Path("cache/hpo")
    ):
        """
        Initialize adaptive grid.

        Args:
            config: HPO configuration
            cache_dir: Directory to cache best parameters
        """
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Default XGBoost parameter ranges (global search)
        self.global_param_ranges = {
            'n_estimators': (300, 1000),
            'learning_rate': (0.01, 0.1),
            'max_depth': (3, 8),
            'subsample': (0.5, 0.95),
            'colsample_bytree': (0.5, 0.95),
            'gamma': (0.0, 2.0),
            'reg_alpha': (0.0, 5.0),
            'reg_lambda': (0.5, 5.0),
            'min_child_weight': (1, 50)
        }

        # Default best parameters (used if no cache found)
        self.default_params = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.5,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'min_child_weight': 10
        }

    def _get_cache_path(self, model_id: str) -> Path:
        """Get path to parameter cache file."""
        return self.cache_dir / f"{model_id}_best_params.json"

    def load_best_params(self, model_id: str) -> Dict[str, Any]:
        """
        Load best parameters from cache.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Best parameters dictionary
        """
        cache_path = self._get_cache_path(model_id)

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                logger.info(f"Loaded best params from cache for {model_id}")
                return cached['best_params']
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Failed to load cached params for {model_id}, using defaults")

        return self.default_params.copy()

    def save_best_params(
        self,
        model_id: str,
        params: Dict[str, Any],
        score: float
    ):
        """
        Save best parameters to cache.

        Args:
            model_id: Unique identifier for the model
            params: Best parameters
            score: Best score achieved
        """
        cache_path = self._get_cache_path(model_id)

        cache_data = {
            'best_params': params,
            'best_score': score,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Saved best params for {model_id} with score {score:.4f}")

    def should_do_global_search(self, model_id: str) -> bool:
        """
        Determine if we should do a global search.

        Args:
            model_id: Unique identifier for the model

        Returns:
            True if should do global search
        """
        cache_path = self._get_cache_path(model_id)

        # First run always does global search
        if not cache_path.exists():
            return True

        # Load run counter
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            run_counter = cached.get('run_counter', 0)
        except (json.JSONDecodeError, KeyError):
            return True

        # Check if it's time for global search
        return (run_counter % self.config.global_every_n_runs) == 0

    def increment_run_counter(self, model_id: str):
        """Increment the run counter for this model."""
        cache_path = self._get_cache_path(model_id)

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                cached['run_counter'] = cached.get('run_counter', 0) + 1
                with open(cache_path, 'w') as f:
                    json.dump(cached, f, indent=2)
            except (json.JSONDecodeError, KeyError):
                pass

    def _create_local_search_space(
        self,
        trial: 'optuna.Trial',
        current_best: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create search space around current best parameters.

        Args:
            trial: Optuna trial
            current_best: Current best parameters

        Returns:
            Parameters dictionary for this trial
        """
        params = {}
        radius = self.config.local_search_radius

        # n_estimators: integer with local range
        center = current_best['n_estimators']
        low = max(300, int(center * (1 - radius)))
        high = min(1000, int(center * (1 + radius)))
        params['n_estimators'] = trial.suggest_int('n_estimators', low, high)

        # learning_rate: log scale with local range
        center = current_best['learning_rate']
        low = max(0.001, center * (1 - radius))
        high = min(0.2, center * (1 + radius))
        params['learning_rate'] = trial.suggest_float('learning_rate', low, high, log=True)

        # max_depth: integer with local range
        center = current_best['max_depth']
        low = max(2, center - 2)
        high = min(10, center + 2)
        params['max_depth'] = trial.suggest_int('max_depth', low, high)

        # subsample: float with local range
        center = current_best['subsample']
        low = max(0.4, center - radius)
        high = min(1.0, center + radius)
        params['subsample'] = trial.suggest_float('subsample', low, high)

        # colsample_bytree: float with local range
        center = current_best['colsample_bytree']
        low = max(0.4, center - radius)
        high = min(1.0, center + radius)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', low, high)

        # gamma: float with local range
        center = current_best['gamma']
        low = max(0.0, center - radius)
        high = min(3.0, center + radius)
        params['gamma'] = trial.suggest_float('gamma', low, high)

        # reg_alpha: float with local range
        center = current_best['reg_alpha']
        low = max(0.0, center - radius * 2)
        high = min(10.0, center + radius * 2)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', low, high)

        # reg_lambda: float with local range
        center = current_best['reg_lambda']
        low = max(0.0, center - radius * 2)
        high = min(10.0, center + radius * 2)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', low, high)

        # min_child_weight: integer with local range
        center = current_best['min_child_weight']
        low = max(1, int(center * (1 - radius)))
        high = min(100, int(center * (1 + radius)))
        params['min_child_weight'] = trial.suggest_int('min_child_weight', low, high)

        return params

    def _create_global_search_space(
        self,
        trial: 'optuna.Trial'
    ) -> Dict[str, Any]:
        """
        Create wide search space for global optimization.

        Args:
            trial: Optuna trial

        Returns:
            Parameters dictionary for this trial
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *self.global_param_ranges['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *self.global_param_ranges['learning_rate'], log=True),
            'max_depth': trial.suggest_int('max_depth', *self.global_param_ranges['max_depth']),
            'subsample': trial.suggest_float('subsample', *self.global_param_ranges['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *self.global_param_ranges['colsample_bytree']),
            'gamma': trial.suggest_float('gamma', *self.global_param_ranges['gamma']),
            'reg_alpha': trial.suggest_float('reg_alpha', *self.global_param_ranges['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *self.global_param_ranges['reg_lambda']),
            'min_child_weight': trial.suggest_int('min_child_weight', *self.global_param_ranges['min_child_weight'])
        }
        return params

    def optimize(
        self,
        model_id: str,
        objective_func: Callable[[Dict[str, Any]], float],
        force_global: bool = False
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters using adaptive grid search.

        Args:
            model_id: Unique identifier for the model
            objective_func: Function that takes params dict and returns score (higher is better)
            force_global: Force global search regardless of counter

        Returns:
            Tuple of (best_params, best_score)
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self.default_params.copy(), 0.0

        # Determine search type
        do_global = force_global or self.should_do_global_search(model_id)

        if do_global:
            logger.info(f"Performing GLOBAL search for {model_id}")
            n_trials = self.config.global_n_trials
        else:
            logger.info(f"Performing LOCAL search for {model_id}")
            n_trials = self.config.local_n_trials

        # Load current best parameters
        current_best = self.load_best_params(model_id)

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        def optuna_objective(trial):
            """Optuna objective function."""
            if do_global:
                params = self._create_global_search_space(trial)
            else:
                params = self._create_local_search_space(trial, current_best)

            try:
                score = objective_func(params)
                return score
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return -np.inf

        # Run optimization
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

        # Get best parameters and score
        best_params = study.best_params
        best_score = study.best_value

        # Save to cache
        self.save_best_params(model_id, best_params, best_score)
        self.increment_run_counter(model_id)

        logger.info(
            f"HPO completed for {model_id}: "
            f"{'GLOBAL' if do_global else 'LOCAL'} search, "
            f"best score = {best_score:.4f}"
        )

        return best_params, best_score


# Import pandas for timestamp
import pandas as pd
