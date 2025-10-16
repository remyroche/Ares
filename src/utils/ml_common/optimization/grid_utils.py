"""
Grid Utilities for HPO

Centralized creation of coarse and fine parameter grids used across HPO
to avoid duplication and ensure consistent behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np

def build_coarse_grid_from_search_space(search_space: Dict[str, Any], grid_points: int) -> List[Dict[str, Any]]:
    """Create a coarse parameter grid list from a search space.

    Each parameter produces a list of candidate values; we then take the Cartesian
    product to return a list of parameter dictionaries.
    """
    try:
        value_lists: List[List[Tuple[str, Any]]] = []
        for name, cfg in search_space.items():
            if not isinstance(cfg, dict):
                # Legacy tuple(low, high)
                if isinstance(cfg, tuple) and len(cfg) == 2:
                    low, high = cfg
                    vals = np.linspace(low, high, num=max(2, grid_points)).tolist()
                    value_lists.append([(name, v) for v in vals])
                continue

            typ = cfg.get('type', 'float')
            if typ == 'float':
                low, high = cfg['low'], cfg['high']
                vals = np.linspace(low, high, num=max(2, grid_points)).tolist()
                value_lists.append([(name, v) for v in vals])
            elif typ == 'int':
                low, high = cfg['low'], cfg['high']
                if high == low:
                    vals = [low]
                else:
                    pts = np.linspace(low, high, num=max(2, grid_points))
                    vals = sorted({int(round(v)) for v in pts})
                value_lists.append([(name, v) for v in vals])
            elif typ == 'categorical':
                vals = cfg.get('choices', [])
                value_lists.append([(name, v) for v in vals])

        if not value_lists:
            return []

        combinations = list(itertools.product(*value_lists))
        return [dict(combo) for combo in combinations]
    except Exception:
        return []

def build_fine_grid_around_best(search_space: Dict[str, Any], best_params: Dict[str, Any],
                                grid_points: int) -> List[Dict[str, Any]]:
    """Create a fine parameter grid around the best parameters discovered so far.

    For floats: +/- 20% of the original range; for ints: +/- 2; categorical: keep choices.
    """
    combos: List[List[Tuple[str, Any]]] = []
    for name, cfg in search_space.items():
        if name not in best_params:
            continue
        best_val = best_params[name]
        if isinstance(cfg, dict):
            typ = cfg.get('type', 'float')
            if typ == 'float':
                low, high = cfg['low'], cfg['high']
                rng = high - low
                fine_rng = rng * 0.2
                fine_min = max(low, best_val - fine_rng)
                fine_max = min(high, best_val + fine_rng)
                if cfg.get('log', False) and fine_min > 0 and fine_max > fine_min:
                    vals = np.logspace(np.log10(fine_min), np.log10(fine_max), grid_points)
                else:
                    vals = np.linspace(fine_min, fine_max, grid_points)
                combos.append([(name, v) for v in vals])
            elif typ == 'int':
                low, high = cfg['low'], cfg['high']
                fine_min = max(low, int(best_val) - 2)
                fine_max = min(high, int(best_val) + 2)
                vals = list(range(fine_min, fine_max + 1))
                combos.append([(name, v) for v in vals])
            elif typ == 'categorical':
                vals = cfg.get('choices', [])
                combos.append([(name, v) for v in vals])
        else:
            # Legacy tuple
            if isinstance(cfg, tuple) and len(cfg) == 2:
                low, high = cfg
                rng = high - low
                fine_rng = rng * 0.2
                fine_min = max(low, best_val - fine_rng)
                fine_max = min(high, best_val + fine_rng)
                vals = np.linspace(fine_min, fine_max, grid_points)
                combos.append([(name, v) for v in vals])

    if not combos:
        return []
    return [dict(c) for c in itertools.product(*combos)]

def generate_grid(search_space: Dict[str, Any], max_trials: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate a parameter grid from a search space.

    This helper leverages :func:`build_coarse_grid_from_search_space` to produce
    cartesian products of parameters. When ``max_trials`` is provided, the helper
    adapts the number of grid points per parameter to stay close to the desired
    number of trials and truncates any excess combinations.

    Args:
        search_space: Configuration describing the search space for each
            parameter.
        max_trials: Optional cap on the number of combinations to return.

    Returns:
        A list of parameter dictionaries representing combinations to explore.
    """

    if not search_space:
        return []

    # Determine a reasonable number of grid points per parameter. For discrete
    # parameters this keeps all values, while for continuous ranges it limits the
    # cartesian product to roughly ``max_trials`` combinations when specified.
    if max_trials is None or max_trials <= 0:
        grid_points = 3
    else:
        param_count = max(len(search_space), 1)
        grid_points = int(np.ceil(max_trials ** (1.0 / param_count)))
        grid_points = max(grid_points, 2)

    combinations = build_coarse_grid_from_search_space(search_space, grid_points)

    if max_trials is not None and max_trials > 0:
        return combinations[:max_trials]
    return combinations

class GridSearchOptimizer:
    """Grid search optimizer for hyperparameter tuning."""

    def __init__(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        scoring: str = 'accuracy',
        cv: int = 5
    ):
        """Initialize grid search optimizer.

        Args:
            param_grid: Dictionary of parameter grids to search
            scoring: Scoring metric to optimize
            cv: Number of cross-validation folds
        """
        self.param_grid = param_grid or {}
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.grid_search_ = None

    def fit(self, X, y, estimator=None, param_grid: Optional[Dict[str, List[Any]]] = None):
        """Fit the grid search optimizer.

        Args:
            X: Feature matrix
            y: Target vector
            estimator: Base estimator to optimize (if None, uses a simple classifier)
            param_grid: Optional parameter grid to use for this fit call
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        if param_grid is not None:
            self.param_grid = param_grid

        if not self.param_grid:
            raise ValueError("param_grid must be provided before calling fit")

        # Default estimator if none provided
        if estimator is None:
            estimator = RandomForestClassifier(random_state=42)

        # Run grid search
        self.grid_search_ = GridSearchCV(
            estimator,
            self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )

        self.grid_search_.fit(X, y)

        self.best_params_ = self.grid_search_.best_params_
        self.best_score_ = self.grid_search_.best_score_
        self.cv_results_ = self.grid_search_.cv_results_

        return self

    def generate_grid(self, search_space: Dict[str, Any], max_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate a parameter grid using the shared helper utility."""

        return generate_grid(search_space, max_trials)

    def predict(self, X):
        """Make predictions using the best estimator found."""
        if self.grid_search_ is None:
            raise ValueError("GridSearchOptimizer must be fitted before making predictions")
        return self.grid_search_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities using the best estimator found."""
        if self.grid_search_ is None:
            raise ValueError("GridSearchOptimizer must be fitted before making predictions")
        if not hasattr(self.grid_search_, 'predict_proba'):
            raise AttributeError("The best estimator does not support predict_proba")
        return self.grid_search_.predict_proba(X)

    def score(self, X, y):
        """Return the score of the best estimator on the given test data."""
        if self.grid_search_ is None:
            raise ValueError("GridSearchOptimizer must be fitted before scoring")
        return self.grid_search_.score(X, y)

    def get_best_params(self):
        """Get the best parameters found."""
        return self.best_params_

    def get_best_score(self):
        """Get the best score achieved."""
        return self.best_score_

    def get_cv_results(self):
        """Get the cross-validation results."""
        return self.cv_results_

    def get_best_estimator(self):
        """Get the best estimator found."""
        if self.grid_search_ is None:
            raise ValueError("GridSearchOptimizer must be fitted before getting best estimator")
        return self.grid_search_.best_estimator_

__all__ = [
    'build_coarse_grid_from_search_space',
    'build_fine_grid_around_best',
    'generate_grid',
    'GridSearchOptimizer',
]
