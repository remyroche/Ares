# src/training/steps/step17_final_parameters_optimization/
# hyperparameter_optimization_config.py

"""Hyperparameter Optimization Configuration.

This module defines comprehensive search spaces, optimization strategies, and evaluation
metrics for Step 12: Final Parameters Optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptimizationStrategy(Enum):
    """Optimization strategies for different parameter categories."""

    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"


class EvaluationMetric(Enum):
    """Evaluation metrics for optimization."""

    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    TOTAL_RETURN = "total_return"
    VOLATILITY = "volatility"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"


@dataclass
class SearchSpace:
    """Defines the search space for a parameter category."""

    name: str = ""
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.SINGLE_OBJECTIVE
    n_trials: int = 50
    timeout_seconds: int = 1800
    early_stopping_patience: int = 10
    evaluation_metrics: list[EvaluationMetric] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    warm_start: bool = True
    parallel_trials: int = 1


@dataclass
class ConfidenceThresholdsSearchSpace(SearchSpace):
    """Search space for confidence thresholds optimization."""


@dataclass
class VolatilityParametersSearchSpace(SearchSpace):
    """Search space for volatility parameters optimization."""


@dataclass
class PositionSizingSearchSpace(SearchSpace):
    """Search space for position sizing parameters optimization."""


@dataclass
class RiskManagementSearchSpace(SearchSpace):
    """Search space for risk management parameters optimization."""


@dataclass
class EnsembleParametersSearchSpace(SearchSpace):
    """Search space for ensemble parameters optimization."""


@dataclass
class RegimeSpecificSearchSpace(SearchSpace):
    """Search space for regime-specific parameters optimization."""


@dataclass
class TimingParametersSearchSpace(SearchSpace):
    """Search space for timing parameters optimization."""


class HyperparameterOptimizationConfig:
    """Main configuration class for hyperparameter optimization."""

    def __init__(self) -> None:
        self.search_spaces: dict[str, SearchSpace] = {
            "confidence_thresholds": ConfidenceThresholdsSearchSpace(),
            "volatility_parameters": VolatilityParametersSearchSpace(),
            "position_sizing_parameters": PositionSizingSearchSpace(),
            "risk_management_parameters": RiskManagementSearchSpace(),
            "ensemble_parameters": EnsembleParametersSearchSpace(),
            "regime_specific_parameters": RegimeSpecificSearchSpace(),
            "timing_parameters": TimingParametersSearchSpace(),
        }

        self.global_config: dict[str, Any] = {
            "storage_url": "sqlite:///data/optimization_storage/optuna_studies.db",
            "study_name_prefix": "hyperparameter_optimization",
            "sampler": "tpe",  # "tpe", "random", "cmaes", "nsgaii"
            "pruner": "hyperband",  # "hyperband", "median", "percentile"
            "n_jobs": -1,  # Number of parallel jobs
            "seed": 42,
            "enable_logging": True,
            "log_level": "INFO",
        }

        self.evaluation_config: dict[str, Any] = {
            "backtest_window_days": 30,
            "validation_window_days": 7,
            "min_trades_for_evaluation": 10,
            "evaluation_metrics": [
                "win_rate",
                "profit_factor",
                "sharpe_ratio",
                "max_drawdown",
                "total_return",
            ],
            "primary_metric": "sharpe_ratio",
            "constraint_metrics": {
                "max_drawdown": {"max": 0.25},
                "win_rate": {"min": 0.4},
                "profit_factor": {"min": 1.2},
            },
        }

    def validate_search_space(self, search_space: SearchSpace) -> list[str]:
        """Validate a search space configuration."""
        errors: list[str] = []

        # Check required fields
        if not search_space.name:
            errors.append("Search space name is required")

        if not search_space.parameters:
            errors.append("Search space parameters are required")

        # Check parameter definitions
        for param_name, param_config in search_space.parameters.items():
            if "type" not in param_config:
                errors.append(f"Parameter {param_name} missing type definition")
                continue

            param_type = param_config.get("type")
            if param_type == "float":
                if "min" not in param_config or "max" not in param_config:
                    errors.append(
                        f"Float parameter {param_name} missing min/max values",
                    )
            elif param_type == "int":
                if "min" not in param_config or "max" not in param_config:
                    errors.append(f"Int parameter {param_name} missing min/max values")
            elif param_type == "categorical":
                if "choices" not in param_config:
                    errors.append(f"Categorical parameter {param_name} missing choices")

        return errors


# Global configuration instance
HYPERPARAMETER_CONFIG = HyperparameterOptimizationConfig()



def validate_hyperparameter_config() -> list[str]:
    """Validate the entire hyperparameter optimization configuration."""
    config = get_hyperparameter_config()
    errors: list[str] = []

    # Validate each search space
    for name, search_space in config.search_spaces.items():
        space_errors = config.validate_search_space(search_space)
        for err in space_errors:
            errors.append(f"{name}: {err}")

    # Validate global config
    if not config.global_config.get("storage_url"):
        errors.append("Global config missing storage_url")

    if not config.global_config.get("study_name_prefix"):
        errors.append("Global config missing study_name_prefix")

    return errors



if __name__ == "__main__":
    # Test the configuration
    config = get_hyperparameter_config()

    # Validate configuration
    errors = validate_hyperparameter_config()
    if errors:
        print("❌ Configuration validation errors:")
        for _error in errors:
            print(f" - {_error}")
    else:
        print("✅ Configuration validated successfully")

    # Print optimization plan
    plan = get_optimization_plan()
    print("\nOptimization plan summary:")
    print(
        f" - Total trials: {plan['summary']['total_trials']} | "
        f"Estimated time (hrs): "
        f"{plan['optimization_plan']['total_estimated_time_hours']:.1f} | "
        f"Parallel: {plan['optimization_plan']['parallel_execution']}"
    )

    # Print search spaces
    print("\nSearch spaces:")
    for _name, _space in config.search_spaces.items():
        print(
            f" - {_name}: parameters={len(_space.parameters)} | "
            f"trials={_space.n_trials} | "
            f"strategy={_space.optimization_strategy.value}"
        )