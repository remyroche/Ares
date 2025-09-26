"""Modular hyperparameter optimisation utilities."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

_LOGGER = logging.getLogger(__name__)

try:  # Optional dependency
    import optuna
    from optuna.samplers import TPESampler
    _OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optuna is optional
    optuna = None  # type: ignore
    TPESampler = object  # type: ignore
    _OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedKFold, TimeSeriesSplit
    from sklearn.metrics import get_scorer
    from sklearn.base import clone
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ParameterGrid = ParameterSampler = StratifiedKFold = TimeSeriesSplit = get_scorer = clone = None  # type: ignore
    _SKLEARN_AVAILABLE = False


@dataclass
class HPOResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": self.trials,
            "metadata": self.metadata,
        }


class HyperparameterOptimization:
    """Lightweight orchestrator for hyperparameter optimisation strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = _LOGGER.getChild("HyperparameterOptimization")
        self.random_state = self.config.get("random_state", 42)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def bayesian_optimization(
        self,
        model_factory: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        scoring: Union[str, Callable] = "accuracy",
        cv: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        if not _OPTUNA_AVAILABLE:
            self.logger.warning("Optuna unavailable – falling back to random search")
            return self.random_search(
                model_factory,
                X,
                y,
                search_space,
                n_trials,
                scoring=scoring,
                cv=cv,
                **fit_kwargs,
            )

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.random_state))

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, bounds in search_space.items():
                if isinstance(bounds, dict) and {"low", "high"} <= bounds.keys():
                    params[name] = trial.suggest_float(name, bounds["low"], bounds["high"], log=bounds.get("log", False))
                elif isinstance(bounds, list):
                    params[name] = trial.suggest_categorical(name, bounds)
                else:
                    params[name] = bounds
            score = self._evaluate(model_factory, params, X, y, scoring, cv, **fit_kwargs)
            return float(score)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        trials = [
            {"params": t.params, "score": t.value, "number": t.number}
            for t in study.trials
        ]
        return HPOResult(study.best_params, float(study.best_value), trials).to_dict()

    def random_search(
        self,
        model_factory: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        scoring: Union[str, Callable] = "accuracy",
        cv: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for random search")

        sampler = ParameterSampler(search_space, n_iter=n_trials, random_state=self.random_state)
        trials: List[Dict[str, Any]] = []
        best_score = -math.inf
        best_params: Dict[str, Any] = {}

        for params in sampler:
            score = self._evaluate(model_factory, params, X, y, scoring, cv, **fit_kwargs)
            trials.append({"params": dict(params), "score": score})
            if score > best_score:
                best_score = score
                best_params = dict(params)

        return HPOResult(best_params, float(best_score), trials, {"method": "random"}).to_dict()

    def grid_search(
        self,
        model_factory: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        scoring: Union[str, Callable] = "accuracy",
        cv: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for grid search")

        grid = ParameterGrid(search_space)
        trials: List[Dict[str, Any]] = []
        best_score = -math.inf
        best_params: Dict[str, Any] = {}

        for params in grid:
            score = self._evaluate(model_factory, params, X, y, scoring, cv, **fit_kwargs)
            trials.append({"params": dict(params), "score": score})
            if score > best_score:
                best_score = score
                best_params = dict(params)

        return HPOResult(best_params, float(best_score), trials, {"method": "grid"}).to_dict()

    def staged_hpo(
        self,
        model_factory: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        bayes_n_trials: int = 30,
        coarse_grid: Optional[Dict[str, Any]] = None,
        scoring: Union[str, Callable] = "accuracy",
        cv: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        coarse = coarse_grid or {key: values[: min(3, len(values))] for key, values in search_space.items() if isinstance(values, list)}
        if coarse:
            grid_result = self.grid_search(model_factory, X, y, coarse, scoring=scoring, cv=cv, **fit_kwargs)
            warm_start = grid_result["best_params"]
        else:
            warm_start = {}

        refined_space = self._refine_search_space(search_space, warm_start)
        return self.bayesian_optimization(
            model_factory,
            X,
            y,
            refined_space,
            n_trials=bayes_n_trials,
            scoring=scoring,
            cv=cv,
            **fit_kwargs,
        )

    def multi_objective_optimization(
        self,
        model_factory: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        objectives: Dict[str, Callable[[Any, np.ndarray, np.ndarray], float]],
        search_space: Dict[str, Any],
        n_trials: int = 40,
        cv: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        weights = self.config.get("objective_weights") or {name: 1.0 for name in objectives}
        trials: List[Dict[str, Any]] = []
        best_params: Dict[str, Any] = {}
        best_score = -math.inf

        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for multi-objective optimisation")

        sampler = ParameterSampler(search_space, n_iter=n_trials, random_state=self.random_state)
        for params in sampler:
            model = model_factory(**params)
            model.fit(X, y, **fit_kwargs)
            score_components = {name: func(model, X, y) for name, func in objectives.items()}
            weighted_score = float(sum(score_components[name] * weights.get(name, 1.0) for name in score_components))
            trials.append({"params": dict(params), "score": weighted_score, "components": score_components})
            if weighted_score > best_score:
                best_score = weighted_score
                best_params = dict(params)

        metadata = {"objectives": list(objectives.keys()), "weights": weights, "method": "multi-objective"}
        return HPOResult(best_params, best_score, trials, metadata).to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        model_factory: Callable[..., Any],
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: Union[str, Callable],
        cv: Optional[Any],
        **fit_kwargs: Any,
    ) -> float:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for scoring utilities")

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        model = model_factory(**params)
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            if len(np.unique(y_arr)) > 20:
                cv = TimeSeriesSplit(n_splits=5)

        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
        scores = []
        for train_idx, test_idx in cv.split(X_arr, y_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            cloned = clone(model)
            cloned.fit(X_train, y_train, **fit_kwargs)
            scores.append(float(scorer(cloned, X_test, y_test)))  # type: ignore[arg-type]
        return float(np.mean(scores))

    def _refine_search_space(self, search_space: Dict[str, Any], warm_start: Dict[str, Any]) -> Dict[str, Any]:
        refined: Dict[str, Any] = {}
        for name, bounds in search_space.items():
            if name not in warm_start:
                refined[name] = bounds
                continue
            value = warm_start[name]
            if isinstance(bounds, dict) and {"low", "high"} <= bounds.keys():
                span = bounds["high"] - bounds["low"]
                refined[name] = {
                    "low": max(bounds["low"], value - span * 0.25),
                    "high": min(bounds["high"], value + span * 0.25),
                    "log": bounds.get("log", False),
                }
            else:
                refined[name] = bounds
        return refined


HyperparameterOptimizer = HyperparameterOptimization


# ---------------------------------------------------------------------------
# Helper functions mirroring the previous public API
# ---------------------------------------------------------------------------


def optimize_hyperparameters(
    model_factory: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    search_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 50,
    method: str = "bayesian",
    scoring: Union[str, Callable] = "accuracy",
    cv: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    hpo = HyperparameterOptimization(config)
    search = search_space or create_search_space(kwargs.get("model_type", "generic"), X, y)

    if method == "bayesian":
        return hpo.bayesian_optimization(model_factory, X, y, search, n_trials=n_trials, scoring=scoring, cv=cv, **kwargs)
    if method == "random":
        return hpo.random_search(model_factory, X, y, search, n_trials=n_trials, scoring=scoring, cv=cv, **kwargs)
    if method == "grid":
        return hpo.grid_search(model_factory, X, y, search, scoring=scoring, cv=cv, **kwargs)
    if method == "staged":
        return hpo.staged_hpo(model_factory, X, y, search, bayes_n_trials=n_trials, scoring=scoring, cv=cv, **kwargs)
    raise ValueError(f"Unknown optimisation method '{method}'")


def bayesian_optimization(
    model_factory: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    search_space: Dict[str, Any],
    *,
    n_trials: int = 50,
    scoring: Union[str, Callable] = "accuracy",
    cv: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return HyperparameterOptimization(config).bayesian_optimization(
        model_factory,
        X,
        y,
        search_space,
        n_trials=n_trials,
        scoring=scoring,
        cv=cv,
        **kwargs,
    )


def grid_search(
    model_factory: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    search_space: Dict[str, Any],
    *,
    scoring: Union[str, Callable] = "accuracy",
    cv: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return HyperparameterOptimization(config).grid_search(
        model_factory,
        X,
        y,
        search_space,
        scoring=scoring,
        cv=cv,
        **kwargs,
    )


def random_search(
    model_factory: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    search_space: Dict[str, Any],
    *,
    n_trials: int = 50,
    scoring: Union[str, Callable] = "accuracy",
    cv: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return HyperparameterOptimization(config).random_search(
        model_factory,
        X,
        y,
        search_space,
        n_trials=n_trials,
        scoring=scoring,
        cv=cv,
        **kwargs,
    )


def staged_hpo(
    model_factory: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    search_space: Dict[str, Any],
    *,
    bayes_n_trials: int = 30,
    scoring: Union[str, Callable] = "accuracy",
    cv: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return HyperparameterOptimization(config).staged_hpo(
        model_factory,
        X,
        y,
        search_space,
        bayes_n_trials=bayes_n_trials,
        scoring=scoring,
        cv=cv,
        **kwargs,
    )


def create_search_space(
    model_type: str,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    data_characteristics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    heuristics = {
        "lightgbm": {
            "learning_rate": {"low": 1e-3, "high": 0.3, "log": True},
            "num_leaves": {"low": 16, "high": 256},
            "min_child_samples": {"low": 5, "high": 200},
        },
        "xgboost": {
            "eta": {"low": 1e-3, "high": 0.3, "log": True},
            "max_depth": {"low": 3, "high": 10},
            "subsample": {"low": 0.5, "high": 1.0},
        },
        "random_forest": {
            "n_estimators": {"low": 100, "high": 800},
            "max_depth": {"low": 3, "high": 20},
            "max_features": ["sqrt", "log2", None],
        },
        "logistic_regression": {
            "C": {"low": 1e-3, "high": 10.0, "log": True},
            "penalty": ["l2", "l1"],
        },
    }

    if data_characteristics is None and X is not None:
        unique_targets = len(np.unique(y)) if y is not None else 2
        data_characteristics = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1] if X.ndim > 1 else 1,
            "n_classes": unique_targets,
        }

    base = heuristics.get(model_type.lower(), {
        "learning_rate": {"low": 1e-4, "high": 0.3, "log": True},
        "max_depth": {"low": 3, "high": 12},
    })

    if data_characteristics:
        base.setdefault("n_estimators", {"low": 100, "high": 1000})
        if data_characteristics.get("n_features", 0) > 100:
            base.setdefault("max_features", ["sqrt", "log2"])

    return base


def validate_hpo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    result = {"valid": True, "warnings": [], "errors": []}
    if "n_trials" in config and (not isinstance(config["n_trials"], int) or config["n_trials"] <= 0):
        result["errors"].append("'n_trials' must be a positive integer")
    if "method" in config and config["method"] not in {"bayesian", "random", "grid", "staged"}:
        result["errors"].append("Unsupported optimisation method")
    if result["errors"]:
        result["valid"] = False
    return result


__all__ = [
    "HPOResult",
    "HyperparameterOptimization",
    "HyperparameterOptimizer",
    "bayesian_optimization",
    "create_search_space",
    "grid_search",
    "optimize_hyperparameters",
    "random_search",
    "staged_hpo",
    "validate_hpo_config",
]
