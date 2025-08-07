# src/training/steps/step12_final_parameters_optimization/hyperparameter_optimization_config.py

"""
Hyperparameter Optimization Configuration

This module defines comprehensive search spaces, optimization strategies, and evaluation
metrics for Step 12: Final Parameters Optimization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


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


@dataclass
class SearchSpace:
    """Defines the search space for a parameter category."""
    
    name: str
    parameters: Dict[str, Dict[str, Any]]
    optimization_strategy: OptimizationStrategy
    n_trials: int = 50
    timeout_seconds: int = 1800
    early_stopping_patience: int = 10
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    warm_start: bool = True
    parallel_trials: int = 1


@dataclass
class ConfidenceThresholdsSearchSpace(SearchSpace):
    """Search space for confidence thresholds optimization."""
    
    def __post_init__(self):
        self.name = "confidence_thresholds"
        self.optimization_strategy = OptimizationStrategy.MULTI_OBJECTIVE
        self.n_trials = 100
        self.timeout_seconds = 1800
        self.evaluation_metrics = [
            EvaluationMetric.WIN_RATE,
            EvaluationMetric.PROFIT_FACTOR,
            EvaluationMetric.SHARPE_RATIO,
            EvaluationMetric.MAX_DRAWDOWN
        ]
        
        self.parameters = {
            "analyst_confidence_threshold": {
                "type": "float",
                "min": 0.5,
                "max": 0.95,
                "step": 0.02,
                "log": False
            },
            "tactician_confidence_threshold": {
                "type": "float",
                "min": 0.5,
                "max": 0.95,
                "step": 0.02,
                "log": False
            },
            "ensemble_confidence_threshold": {
                "type": "float",
                "min": 0.5,
                "max": 0.95,
                "step": 0.02,
                "log": False
            },
            "position_scale_up_threshold": {
                "type": "float",
                "min": 0.7,
                "max": 0.95,
                "step": 0.02,
                "log": False
            },
            "position_scale_down_threshold": {
                "type": "float",
                "min": 0.4,
                "max": 0.7,
                "step": 0.02,
                "log": False
            },
            "position_close_threshold": {
                "type": "float",
                "min": 0.2,
                "max": 0.5,
                "step": 0.02,
                "log": False
            },
            "ml_target_update_threshold": {
                "type": "float",
                "min": 0.3,
                "max": 0.7,
                "step": 0.05,
                "log": False
            },
            "emergency_update_threshold": {
                "type": "float",
                "min": 0.01,
                "max": 0.05,
                "step": 0.005,
                "log": False
            }
        }
        
        # Constraints
        self.constraints = {
            "analyst_confidence_threshold": {
                "min": 0.6,
                "max": 0.9
            },
            "tactician_confidence_threshold": {
                "min": 0.55,
                "max": 0.85
            },
            "ensemble_confidence_threshold": {
                "min": 0.65,
                "max": 0.9
            }
        }


@dataclass
class VolatilityParametersSearchSpace(SearchSpace):
    """Search space for volatility parameters optimization."""
    
    def __post_init__(self):
        self.name = "volatility_parameters"
        self.optimization_strategy = OptimizationStrategy.SINGLE_OBJECTIVE
        self.n_trials = 50
        self.evaluation_metrics = [EvaluationMetric.SHARPE_RATIO, EvaluationMetric.VOLATILITY]
        
        self.parameters = {
            "target_volatility": {
                "type": "float",
                "min": 0.05,
                "max": 0.25,
                "step": 0.01,
                "log": False
            },
            "volatility_lookback_period": {
                "type": "int",
                "min": 10,
                "max": 50,
                "step": 5
            },
            "volatility_multiplier": {
                "type": "float",
                "min": 0.5,
                "max": 2.0,
                "step": 0.1,
                "log": False
            },
            "low_volatility_threshold": {
                "type": "float",
                "min": 0.01,
                "max": 0.05,
                "step": 0.005,
                "log": False
            },
            "medium_volatility_threshold": {
                "type": "float",
                "min": 0.03,
                "max": 0.08,
                "step": 0.005,
                "log": False
            },
            "high_volatility_threshold": {
                "type": "float",
                "min": 0.08,
                "max": 0.15,
                "step": 0.01,
                "log": False
            },
            "volatility_stop_loss_multiplier": {
                "type": "float",
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "log": False
            },
            "volatility_based_position_sizing": {
                "type": "categorical",
                "choices": [True, False]
            }
        }


@dataclass
class PositionSizingSearchSpace(SearchSpace):
    """Search space for position sizing parameters optimization."""
    
    def __post_init__(self):
        self.name = "position_sizing_parameters"
        self.optimization_strategy = OptimizationStrategy.SINGLE_OBJECTIVE
        self.n_trials = 60
        self.evaluation_metrics = [EvaluationMetric.TOTAL_RETURN, EvaluationMetric.MAX_DRAWDOWN]
        
        self.parameters = {
            "base_position_size": {
                "type": "float",
                "min": 0.01,
                "max": 0.2,
                "step": 0.01,
                "log": False
            },
            "max_position_size": {
                "type": "float",
                "min": 0.1,
                "max": 0.5,
                "step": 0.05,
                "log": False
            },
            "min_position_size": {
                "type": "float",
                "min": 0.005,
                "max": 0.05,
                "step": 0.005,
                "log": False
            },
            "kelly_multiplier": {
                "type": "float",
                "min": 0.1,
                "max": 0.5,
                "step": 0.05,
                "log": False
            },
            "fractional_kelly": {
                "type": "categorical",
                "choices": [True, False]
            },
            "confidence_based_scaling": {
                "type": "categorical",
                "choices": [True, False]
            },
            "low_confidence_multiplier": {
                "type": "float",
                "min": 0.3,
                "max": 0.8,
                "step": 0.05,
                "log": False
            },
            "medium_confidence_multiplier": {
                "type": "float",
                "min": 0.8,
                "max": 1.2,
                "step": 0.05,
                "log": False
            },
            "high_confidence_multiplier": {
                "type": "float",
                "min": 1.2,
                "max": 2.5,
                "step": 0.1,
                "log": False
            },
            "very_high_confidence_multiplier": {
                "type": "float",
                "min": 1.5,
                "max": 3.0,
                "step": 0.1,
                "log": False
            }
        }
        
        # Constraints
        self.constraints = {
            "max_position_size": {
                "min": 0.05,
                "max": 0.3
            },
            "kelly_multiplier": {
                "min": 0.15,
                "max": 0.4
            }
        }


@dataclass
class RiskManagementSearchSpace(SearchSpace):
    """Search space for risk management parameters optimization."""
    
    def __post_init__(self):
        self.name = "risk_management_parameters"
        self.optimization_strategy = OptimizationStrategy.SINGLE_OBJECTIVE
        self.n_trials = 50
        self.evaluation_metrics = [EvaluationMetric.MAX_DRAWDOWN, EvaluationMetric.VALUE_AT_RISK]
        
        self.parameters = {
            "stop_loss_atr_multiplier": {
                "type": "float",
                "min": 1.0,
                "max": 4.0,
                "step": 0.1,
                "log": False
            },
            "trailing_stop_atr_multiplier": {
                "type": "float",
                "min": 0.8,
                "max": 3.0,
                "step": 0.1,
                "log": False
            },
            "stop_loss_confidence_threshold": {
                "type": "float",
                "min": 0.2,
                "max": 0.5,
                "step": 0.02,
                "log": False
            },
            "enable_dynamic_stop_loss": {
                "type": "categorical",
                "choices": [True, False]
            },
            "volatility_based_sl": {
                "type": "categorical",
                "choices": [True, False]
            },
            "regime_based_sl": {
                "type": "categorical",
                "choices": [True, False]
            },
            "sl_tightening_threshold": {
                "type": "float",
                "min": 0.3,
                "max": 0.6,
                "step": 0.05,
                "log": False
            },
            "sl_loosening_threshold": {
                "type": "float",
                "min": 0.7,
                "max": 0.9,
                "step": 0.05,
                "log": False
            },
            "max_drawdown_threshold": {
                "type": "float",
                "min": 0.1,
                "max": 0.3,
                "step": 0.02,
                "log": False
            },
            "max_daily_loss": {
                "type": "float",
                "min": 0.05,
                "max": 0.15,
                "step": 0.01,
                "log": False
            }
        }


@dataclass
class EnsembleParametersSearchSpace(SearchSpace):
    """Search space for ensemble parameters optimization."""
    
    def __post_init__(self):
        self.name = "ensemble_parameters"
        self.optimization_strategy = OptimizationStrategy.SINGLE_OBJECTIVE
        self.n_trials = 40
        self.evaluation_metrics = [EvaluationMetric.WIN_RATE, EvaluationMetric.PROFIT_FACTOR]
        
        self.parameters = {
            "ensemble_method": {
                "type": "categorical",
                "choices": ["confidence_weighted", "weighted_average", "meta_learner", "majority_vote"]
            },
            "analyst_weight": {
                "type": "float",
                "min": 0.2,
                "max": 0.6,
                "step": 0.05,
                "log": False
            },
            "tactician_weight": {
                "type": "float",
                "min": 0.2,
                "max": 0.6,
                "step": 0.05,
                "log": False
            },
            "strategist_weight": {
                "type": "float",
                "min": 0.1,
                "max": 0.4,
                "step": 0.05,
                "log": False
            },
            "min_ensemble_agreement": {
                "type": "float",
                "min": 0.5,
                "max": 0.8,
                "step": 0.05,
                "log": False
            },
            "max_ensemble_disagreement": {
                "type": "float",
                "min": 0.2,
                "max": 0.5,
                "step": 0.05,
                "log": False
            },
            "ensemble_minimum_models": {
                "type": "int",
                "min": 2,
                "max": 5,
                "step": 1
            }
        }


@dataclass
class RegimeSpecificSearchSpace(SearchSpace):
    """Search space for regime-specific parameters optimization."""
    
    def __post_init__(self):
        self.name = "regime_specific_parameters"
        self.optimization_strategy = OptimizationStrategy.SINGLE_OBJECTIVE
        self.n_trials = 30
        self.evaluation_metrics = [EvaluationMetric.SHARPE_RATIO, EvaluationMetric.TOTAL_RETURN]
        
        self.parameters = {
            "bull_trend_multiplier": {
                "type": "float",
                "min": 0.8,
                "max": 1.5,
                "step": 0.05,
                "log": False
            },
            "bear_trend_multiplier": {
                "type": "float",
                "min": 0.5,
                "max": 1.2,
                "step": 0.05,
                "log": False
            },
            "sideways_multiplier": {
                "type": "float",
                "min": 0.7,
                "max": 1.3,
                "step": 0.05,
                "log": False
            },
            "high_impact_multiplier": {
                "type": "float",
                "min": 0.4,
                "max": 1.0,
                "step": 0.05,
                "log": False
            },
            "sr_zone_multiplier": {
                "type": "float",
                "min": 0.8,
                "max": 1.4,
                "step": 0.05,
                "log": False
            },
            "regime_transition_threshold": {
                "type": "float",
                "min": 0.4,
                "max": 0.8,
                "step": 0.05,
                "log": False
            },
            "regime_confirmation_periods": {
                "type": "int",
                "min": 2,
                "max": 5,
                "step": 1
            }
        }


@dataclass
class TimingParametersSearchSpace(SearchSpace):
    """Search space for timing parameters optimization."""
    
    def __post_init__(self):
        self.name = "timing_parameters"
        self.optimization_strategy = OptimizationStrategy.SINGLE_OBJECTIVE
        self.n_trials = 30
        self.evaluation_metrics = [EvaluationMetric.TOTAL_RETURN, EvaluationMetric.WIN_RATE]
        
        self.parameters = {
            "base_cooldown_minutes": {
                "type": "int",
                "min": 15,
                "max": 60,
                "step": 5
            },
            "high_confidence_cooldown": {
                "type": "int",
                "min": 5,
                "max": 30,
                "step": 5
            },
            "low_confidence_cooldown": {
                "type": "int",
                "min": 30,
                "max": 120,
                "step": 10
            },
            "bull_trend_cooldown": {
                "type": "int",
                "min": 10,
                "max": 40,
                "step": 5
            },
            "bear_trend_cooldown": {
                "type": "int",
                "min": 20,
                "max": 60,
                "step": 5
            },
            "sideways_cooldown": {
                "type": "int",
                "min": 30,
                "max": 90,
                "step": 10
            },
            "high_impact_cooldown": {
                "type": "int",
                "min": 60,
                "max": 180,
                "step": 15
            }
        }


class HyperparameterOptimizationConfig:
    """Main configuration class for hyperparameter optimization."""
    
    def __init__(self):
        self.search_spaces = {
            "confidence_thresholds": ConfidenceThresholdsSearchSpace(),
            "volatility_parameters": VolatilityParametersSearchSpace(),
            "position_sizing_parameters": PositionSizingSearchSpace(),
            "risk_management_parameters": RiskManagementSearchSpace(),
            "ensemble_parameters": EnsembleParametersSearchSpace(),
            "regime_specific_parameters": RegimeSpecificSearchSpace(),
            "timing_parameters": TimingParametersSearchSpace(),
        }
        
        self.global_config = {
            "storage_url": "sqlite:///data/optimization_storage/optuna_studies.db",
            "study_name_prefix": "hyperparameter_optimization",
            "sampler": "tpe",  # "tpe", "random", "cmaes", "nsgaii"
            "pruner": "hyperband",  # "hyperband", "median", "percentile"
            "n_jobs": -1,  # Number of parallel jobs
            "seed": 42,
            "enable_logging": True,
            "log_level": "INFO",
        }
        
        self.evaluation_config = {
            "backtest_window_days": 30,
            "validation_window_days": 7,
            "min_trades_for_evaluation": 10,
            "evaluation_metrics": [
                "win_rate",
                "profit_factor", 
                "sharpe_ratio",
                "max_drawdown",
                "total_return"
            ],
            "primary_metric": "sharpe_ratio",
            "constraint_metrics": {
                "max_drawdown": {"max": 0.25},
                "win_rate": {"min": 0.4},
                "profit_factor": {"min": 1.2}
            }
        }
    
    def get_search_space(self, name: str) -> Optional[SearchSpace]:
        """Get a specific search space by name."""
        return self.search_spaces.get(name)
    
    def get_all_search_spaces(self) -> Dict[str, SearchSpace]:
        """Get all search spaces."""
        return self.search_spaces
    
    def validate_search_space(self, search_space: SearchSpace) -> List[str]:
        """Validate a search space configuration."""
        errors = []
        
        # Check required fields
        if not search_space.name:
            errors.append("Search space name is required")
        
        if not search_space.parameters:
            errors.append("Search space parameters are required")
        
        # Check parameter definitions
        for param_name, param_config in search_space.parameters.items():
            if "type" not in param_config:
                errors.append(f"Parameter {param_name} missing type definition")
            
            param_type = param_config.get("type")
            if param_type == "float":
                if "min" not in param_config or "max" not in param_config:
                    errors.append(f"Float parameter {param_name} missing min/max values")
            elif param_type == "int":
                if "min" not in param_config or "max" not in param_config:
                    errors.append(f"Int parameter {param_name} missing min/max values")
            elif param_type == "categorical":
                if "choices" not in param_config:
                    errors.append(f"Categorical parameter {param_name} missing choices")
        
        return errors
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimization configurations."""
        summary = {
            "total_search_spaces": len(self.search_spaces),
            "total_parameters": sum(
                len(space.parameters) for space in self.search_spaces.values()
            ),
            "total_trials": sum(
                space.n_trials for space in self.search_spaces.values()
            ),
            "search_spaces": {}
        }
        
        for name, space in self.search_spaces.items():
            summary["search_spaces"][name] = {
                "parameters": len(space.parameters),
                "n_trials": space.n_trials,
                "strategy": space.optimization_strategy.value,
                "timeout_seconds": space.timeout_seconds,
                "evaluation_metrics": [metric.value for metric in space.evaluation_metrics]
            }
        
        return summary


# Global configuration instance
HYPERPARAMETER_CONFIG = HyperparameterOptimizationConfig()


def get_hyperparameter_config() -> HyperparameterOptimizationConfig:
    """Get the global hyperparameter optimization configuration."""
    return HYPERPARAMETER_CONFIG


def validate_hyperparameter_config() -> List[str]:
    """Validate the entire hyperparameter optimization configuration."""
    config = get_hyperparameter_config()
    errors = []
    
    # Validate each search space
    for name, search_space in config.search_spaces.items():
        space_errors = config.validate_search_space(search_space)
        for error in space_errors:
            errors.append(f"{name}: {error}")
    
    # Validate global config
    if not config.global_config.get("storage_url"):
        errors.append("Global config missing storage_url")
    
    if not config.global_config.get("study_name_prefix"):
        errors.append("Global config missing study_name_prefix")
    
    return errors


def get_optimization_plan() -> Dict[str, Any]:
    """Get a detailed optimization plan."""
    config = get_hyperparameter_config()
    summary = config.get_optimization_summary()
    
    plan = {
        "optimization_plan": {
            "total_estimated_time_hours": summary["total_trials"] * 0.5 / 3600,  # 30 min per trial
            "total_estimated_cost": summary["total_trials"] * 0.1,  # $0.10 per trial
            "parallel_execution": config.global_config["n_jobs"] > 1,
            "search_spaces_order": list(config.search_spaces.keys()),
            "dependencies": {
                "confidence_thresholds": [],
                "volatility_parameters": ["confidence_thresholds"],
                "position_sizing_parameters": ["confidence_thresholds"],
                "risk_management_parameters": ["confidence_thresholds"],
                "ensemble_parameters": ["confidence_thresholds"],
                "regime_specific_parameters": ["confidence_thresholds"],
                "timing_parameters": ["confidence_thresholds"]
            }
        },
        "summary": summary
    }
    
    return plan


if __name__ == "__main__":
    # Test the configuration
    config = get_hyperparameter_config()
    
    # Validate configuration
    errors = validate_hyperparameter_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid")
    
    # Print optimization plan
    plan = get_optimization_plan()
    print("\nOptimization Plan:")
    print(f"  Total trials: {plan['summary']['total_trials']}")
    print(f"  Estimated time: {plan['optimization_plan']['total_estimated_time_hours']:.1f} hours")
    print(f"  Parallel execution: {plan['optimization_plan']['parallel_execution']}")
    
    # Print search spaces
    print("\nSearch Spaces:")
    for name, space in config.search_spaces.items():
        print(f"  {name}: {len(space.parameters)} parameters, {space.n_trials} trials")