# src/config_optuna.py

"""
Optuna Configuration for Strategy-Level Meta-Parameters

This file contains all the trading parameters that can be optimized during training.
These parameters are used throughout the codebase and should be referenced from this file
instead of being hardcoded in individual components.
"""

from typing import Any
from dataclasses import asdict, dataclass
from enum import Enum


import class EnsembleMethod
class EnsembleMethod(Enum):
    """Enum for ensemble gathering methods."""

    ALL_THRESHOLD = "all_threshold"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    META_LEARNER = "meta_learner"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    REGIME_SPECIFIC = "regime_specific"


class RiskLevel(Enum):
    """Enum for risk levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for different trading decisions."""

    # Entry thresholds
    base_entry_threshold: float = 0.7
    volatility_modulated_entry: bool = True
    volatility_multiplier: float = 0.5
    volatility_zscore_threshold: float = 1.0

    # Analyst vs Tactician thresholds
    analyst_confidence_threshold: float = 0.7
    tactician_confidence_threshold: float = 0.8

    # Position management thresholds
    position_scale_up_threshold: float = 0.85
    position_scale_down_threshold: float = 0.6
    position_close_threshold: float = 0.3

    # ML target update thresholds
    ml_target_update_threshold: float = 0.5
    emergency_update_threshold: float = 0.02

    # Ensemble thresholds
    ensemble_agreement_threshold: float = 0.8
    ensemble_minimum_models: int = 3

    # Position closing thresholds
    neutral_signal_threshold: float = 0.5
    tactician_close_threshold: float = 0.6

    # Model performance thresholds
    model_performance_threshold: float = 0.6
    model_degradation_threshold: float = 0.4
    model_retrain_threshold: float = 0.3

    # Regime-specific thresholds
    bull_trend_threshold: float = 0.65
    bear_trend_threshold: float = 0.75
    sideways_threshold: float = 0.8
    sr_zone_threshold: float = 0.7
    high_impact_candle_threshold: float = 0.9


@dataclass
class VolatilityParameters:
    """Volatility-based parameters for position sizing and risk management."""

    # Volatility targeting
    target_volatility: float = 0.15
    volatility_lookback_period: int = 20
    volatility_multiplier: float = 1.0

    # Volatility thresholds
    low_volatility_threshold: float = 0.02
    medium_volatility_threshold: float = 0.05
    high_volatility_threshold: float = 0.10

    # Volatility-based position sizing
    low_volatility_multiplier: float = 1.2
    medium_volatility_multiplier: float = 1.0
    high_volatility_multiplier: float = 0.7

    # Volatility-based stop losses
    volatility_stop_loss_multiplier: float = 2.0

    # Volatility-based take profits
    volatility_take_profit_multiplier: float = 3.0

    # Volatility regime detection
    volatility_regime_lookback: int = 30
    volatility_regime_threshold: float = 0.02


@dataclass
class EnsembleParameters:
    """Parameters for ensemble model combination."""

    # Ensemble method
    ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE

    # Threshold-based ensemble
    all_threshold_confidence: float = 0.8
    majority_vote_threshold: float = 0.6

    # Weighted ensemble
    analyst_weight: float = 0.4
    tactician_weight: float = 0.3
    strategist_weight: float = 0.3

    # Meta-learner parameters
    meta_learner_type: str = "lightgbm"
    meta_learner_learning_rate: float = 0.1
    meta_learner_n_estimators: int = 100

    # Regime-specific ensemble
    regime_specific_weights: dict[str, float] = None

    # Ensemble validation
    min_ensemble_agreement: float = 0.7
    max_ensemble_disagreement: float = 0.3

    def __post_init__(self):
    pass
    pass
    pass
        if self.regime_specific_weights is None:
    pass
    pass
    pass
            self.regime_specific_weights = {
                "BULL_TREND": 1.2,
                "BEAR_TREND": 0.8,
                "SIDEWAYS_RANGE": 0.9,
                "HIGH_IMPACT_CANDLE": 0.6,
                "SR_ZONE_ACTION": 1.1,
            }


@dataclass
class RiskManagementParameters:
    """Comprehensive risk management parameters."""

    # Portfolio-level risk
    max_portfolio_risk: float = 0.15
    max_correlation_exposure: float = 0.2
    max_sector_exposure: float = 0.3

    # Position-level risk
    max_single_position: float = 0.15
    max_total_exposure: float = 0.3
    max_leverage: float = 10.0

    # Risk metrics
    var_confidence_level: float = 0.95
    max_var_threshold: float = 0.02
    max_cvar_threshold: float = 0.03

    # Dynamic risk adjustment
    enable_dynamic_risk: bool = True
    volatility_scaling: bool = True
    regime_based_risk: bool = True

    # Risk limits
    max_drawdown: float = 0.25
    max_daily_loss: float = 0.1
    max_consecutive_losses: int = 5


@dataclass
class MarketRegimeParameters:
    """Market regime detection and adaptation parameters."""

    # Regime detection
    regime_lookback_period: int = 50
    regime_volatility_threshold: float = 0.02
    regime_trend_threshold: float = 0.01
    regime_stability_threshold: float = 0.7

    # Regime-specific parameters
    bull_trend_multiplier: float = 1.2
    bear_trend_multiplier: float = 0.8
    sideways_multiplier: float = 0.9
    high_impact_multiplier: float = 0.6
    sr_zone_multiplier: float = 1.1

    # Regime transition
    regime_transition_threshold: float = 0.6
    regime_confirmation_periods: int = 3

    # Regime-based optimization
    enable_regime_specific_optimization: bool = True
    regime_specific_constraints: dict[str, dict[str, list[float]]] = None

    def __post_init__(self):
    pass
    pass
    pass
        if self.regime_specific_constraints is None:
    pass
    pass
    pass
            self.regime_specific_constraints = {
                "bull": {
                    "tp_multiplier_range": [2.5, 5.0],
                    "sl_multiplier_range": [1.2, 2.5],
                    "position_size_range": [0.10, 0.25],
                },
                "bear": {
                    "tp_multiplier_range": [2.0, 4.5],
                    "sl_multiplier_range": [1.0, 2.2],
                    "position_size_range": [0.08, 0.20],
                },
                "sideways": {
                    "tp_multiplier_range": [1.5, 3.0],
                    "sl_multiplier_range": [0.8, 1.8],
                    "position_size_range": [0.06, 0.15],
                },
            }


@dataclass
class SROptimizationParameters:
    """
    Comprehensive S/R (Support/Resistance) optimization parameters.

    This dataclass contains all parameters that can be optimized for S/R analysis = including strength score weights, level detection parameters = breakout thresholds,
    zone multipliers = and confidence thresholds.
    """

    # === STRENGTH SCORE WEIGHTS ===
    # Weights for the strength score formula:
    # Strength_score = (w1 * log(Touch Count)) + (w2 * log(Total Volume)) +
    #                  (w3 * log(Level Age)) + (w4 * Bounce Rate) + (w5 * Isolation_Score)
    touch_count_weight: float = 0.3
    total_volume_weight: float = 0.25
    level_age_weight: float = 0.2
    bounce_rate_weight: float = 0.15
    isolation_score_weight: float = 0.1

    # === LEVEL DETECTION PARAMETERS ===
    # Minimum requirements for S/R level identification
    min_touch_count: int = 3
    min_level_age_hours: int = 24
    price_tolerance_pct: float = 0.5
    volume_threshold: float = 1.0
    strength_threshold: float = 0.5

    # === BREAKOUT THRESHOLDS ===
    # Parameters for detecting S/R breakouts
    breakout_threshold: float = 0.75
    confirmation_periods: int = 2
    volume_confirmation: float = 1.5
    momentum_threshold: float = 0.2
    false_breakout_filter: float = 0.2

    # === ZONE MULTIPLIERS ===
    # Multipliers for S/R zone calculations
    support_zone_multiplier: float = 1.0
    resistance_zone_multiplier: float = 1.0
    sr_zone_threshold: float = 0.7
    zone_expansion_factor: float = 1.2
    zone_contraction_factor: float = 0.8

    # === CONFIDENCE THRESHOLDS ===
    # Thresholds for S/R confidence levels
    min_sr_confidence: float = 0.6
    high_confidence_threshold: float = 0.8
    confidence_decay_rate: float = 0.2
    regime_confidence_boost: float = 0.15
    ensemble_confidence_threshold: float = 0.7

    # === OPTIMIZATION CONFIGURATION ===
    # Parameters for the optimization process itself
    multi_objective: bool = True
    objectives: list[str] = None
    objective_weights: dict[str , float] = None

    # Optimization constraints
    n_trials: int = 100
    cv_folds: int = 5
    early_stopping_patience: int = 20
    subsample_fraction: float = 0.7

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.15
    min_win_rate: float = 0.55
    min_profit_factor: float = 1.3
    min_signal_clarity: float = 0.1

    def __post_init__(self):
    pass
    pass
    pass
        if self.objectives is None:
    pass
    pass
    pass
            self.objectives = ["sharpe_ratio", "win_rate", "signal_clarity"]

        if self.objective_weights is None:
    pass
    pass
    pass
            self.objective_weights = {
                "sharpe_ratio": 0.4,
                "win_rate": 0.3,
                "signal_clarity": 0.3,
            }

    def get_strength_score_weights(self) -> dict[str, float]:
    pass
    pass
    pass
        """Get strength score weights as a dictionary."""
        return {
            "touch_count": self.touch_count_weight,
            "total_volume": self.total_volume_weight,
            "level_age": self.level_age_weight,
            "bounce_rate": self.bounce_rate_weight,
            "isolation_score": self.isolation_score_weight,
        }

    def get_level_detection_params(self) -> dict[str, Any]:
    pass
    pass
    pass
        """Get level detection parameters as a dictionary."""
        return {
            "min_touch_count": self.min_touch_count,
            "min_level_age_hours": self.min_level_age_hours,
            "price_tolerance_pct": self.price_tolerance_pct,
            "volume_threshold": self.volume_threshold,
            "strength_threshold": self.strength_threshold,
        }

    def get_breakout_thresholds(self) -> dict[str , float]:
    pass
    pass
    pass
        """Get breakout thresholds as a dictionary."""
        return {
            "breakout_threshold": self.breakout_threshold , "confirmation_periods": self.confirmation_periods,
            "volume_confirmation": self.volume_confirmation , "momentum_threshold": self.momentum_threshold,
            "false_breakout_filter": self.false_breakout_filter,
        }

    def get_zone_multipliers(self) -> dict[str , float]:
    pass
    pass
    pass
        """Get zone multipliers as a dictionary."""
        return {
            "support_zone_multiplier": self.support_zone_multiplier , "resistance_zone_multiplier": self.resistance_zone_multiplier,
            "sr_zone_threshold": self.sr_zone_threshold , "zone_expansion_factor": self.zone_expansion_factor,
            "zone_contraction_factor": self.zone_contraction_factor,
        }

    def get_confidence_thresholds(self) -> dict[str , float]:
    pass
    pass
    pass
        """Get confidence thresholds as a dictionary."""
        return {
            "min_sr_confidence": self.min_sr_confidence , "high_confidence_threshold": self.high_confidence_threshold,
            "confidence_decay_rate": self.confidence_decay_rate , "regime_confidence_boost": self.regime_confidence_boost,
            "ensemble_confidence_threshold": self.ensemble_confidence_threshold,
        }


@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # General optimization settings
    enable_optimization: bool = True
    optimization_method: str = "optuna"  # optuna = grid_search, random_search
    max_trials: int = 100
    timeout_minutes: int = 60

    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, time_series_split

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001

    # Pruning settings
    enable_pruning: bool = True
    pruning_method: str = "hyperband"  # hyperband = median, percentile

    # Multi-objective optimization
    enable_multi_objective: bool = True
    objectives: list[str] = None
    objective_weights: dict[str , float] = None

    # S/R specific optimization
    enable_sr_optimization: bool = True
    sr_optimization_config: SROptimizationParameters = None

    def __post_init__(self):
    pass
    pass
    pass
        if self.objectives is None:
    pass
    pass
    pass
            self.objectives = ["accuracy", "f1_score", "precision"]

        if self.objective_weights is None:
    pass
    pass
    pass
            self.objective_weights = {
                "accuracy": 0.4,
                "f1_score": 0.4,
                "precision": 0.2,
            }

        if self.sr_optimization_config is None:
    pass
    pass
    pass
            self.sr_optimization_config = SROptimizationParameters()


# === GLOBAL CONFIGURATION ===

# Default parameter values
DEFAULT_CONFIDENCE_THRESHOLDS = ConfidenceThresholds()
DEFAULT_VOLATILITY_PARAMETERS = VolatilityParameters()
DEFAULT_ENSEMBLE_PARAMETERS = EnsembleParameters()
DEFAULT_RISK_MANAGEMENT_PARAMETERS = RiskManagementParameters()
DEFAULT_MARKET_REGIME_PARAMETERS = MarketRegimeParameters()
DEFAULT_SR_OPTIMIZATION_PARAMETERS = SROptimizationParameters()
DEFAULT_HYPERPARAMETER_OPTIMIZATION_CONFIG = HyperparameterOptimizationConfig()

# Parameter search spaces for optimization
PARAMETER_SEARCH_SPACES = {
    # Confidence thresholds
    "confidence_thresholds": {
        "base_entry_threshold": {"min": 0.5, "max": 0.9, "type": "float"},
        "analyst_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "tactician_confidence_threshold": {"min": 0.7, "max": 0.95, "type": "float"},
        "sr_zone_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
    },
    # Volatility parameters
    "volatility_parameters": {
        "target_volatility": {"min": 0.1, "max": 0.25, "type": "float"},
        "volatility_lookback_period": {"min": 10, "max": 50, "type": "int"},
        "volatility_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},
    },
    # Ensemble parameters
    "ensemble_parameters": {
        "analyst_weight": {"min": 0.2, "max": 0.6, "type": "float"},
        "tactician_weight": {"min": 0.2, "max": 0.5, "type": "float"},
        "strategist_weight": {"min": 0.2, "max": 0.5, "type": "float"},
    },
    # Risk management parameters
    "risk_management_parameters": {
        "max_portfolio_risk": {"min": 0.1, "max": 0.25, "type": "float"},
        "max_single_position": {"min": 0.1, "max": 0.25, "type": "float"},
        "max_drawdown": {"min": 0.15, "max": 0.35, "type": "float"},
    },
    # Market regime parameters
    "market_regime_parameters": {
        "regime_lookback_period": {"min": 30, "max": 100, "type": "int"},
        "regime_volatility_threshold": {"min": 0.01, "max": 0.05, "type": "float"},
        "regime_trend_threshold": {"min": 0.005, "max": 0.02, "type": "float"},
        "sr_zone_multiplier": {"min": 0.8, "max": 1.5, "type": "float"},
    },
    # S/R optimization parameters
    "sr_optimization_parameters": {
        # Strength score weights
        "touch_count_weight": {"min": 0.1, "max": 0.5, "type": "float"},
        "total_volume_weight": {"min": 0.1, "max": 0.4, "type": "float"},
        "level_age_weight": {"min": 0.1, "max": 0.4, "type": "float"},
        "bounce_rate_weight": {"min": 0.1, "max": 0.4, "type": "float"},
        "isolation_score_weight": {"min": 0.05, "max": 0.3, "type": "float"},
        # Level detection parameters
        "min_touch_count": {"min": 2, "max": 10, "type": "int"},
        "min_level_age_hours": {"min": 1, "max": 48, "type": "int"},
        "price_tolerance_pct": {"min": 0.1, "max": 2.0, "type": "float"},
        "volume_threshold": {"min": 0.5, "max": 2.0, "type": "float"},
        "strength_threshold": {"min": 0.3, "max": 0.8, "type": "float"},
        # Breakout thresholds
        "breakout_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "confirmation_periods": {"min": 1, "max": 5, "type": "int"},
        "volume_confirmation": {"min": 1.2, "max": 3.0, "type": "float"},
        "momentum_threshold": {"min": 0.1, "max": 0.5, "type": "float"},
        "false_breakout_filter": {"min": 0.1, "max": 0.3, "type": "float"},
        # Zone multipliers
        "support_zone_multiplier": {"min": 0.8, "max": 1.5, "type": "float"},
        "resistance_zone_multiplier": {"min": 0.8, "max": 1.5, "type": "float"},
        "sr_zone_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "zone_expansion_factor": {"min": 1.0, "max": 2.0, "type": "float"},
        "zone_contraction_factor": {"min": 0.5, "max": 1.0, "type": "float"},
        # Confidence thresholds
        "min_sr_confidence": {"min": 0.5, "max": 0.8, "type": "float"},
        "high_confidence_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
        "confidence_decay_rate": {"min": 0.1, "max": 0.5, "type": "float"},
        "regime_confidence_boost": {"min": 0.1, "max": 0.3, "type": "float"},
        "ensemble_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
    },
}


def get_parameter_value(param_name: str, default_value: Any = None) -> Any:
    pass
    pass
    pass
    """
    Get parameter value from configuration.

    Args:
        param_name: Name of the parameter
        default_value: Default value if parameter not found

    Returns:
        Parameter value
    """
    # This function can be extended to read from environment variables = # configuration files, or other sources
    return default_value


def get_sr_optimization_config() -> SROptimizationParameters:
    pass
    pass
    pass
    """Get S/R optimization configuration."""
    return DEFAULT_SR_OPTIMIZATION_PARAMETERS


def get_hyperparameter_optimization_config() -> HyperparameterOptimizationConfig:
    pass
    pass
    pass
    """Get hyperparameter optimization configuration."""
    return DEFAULT_HYPERPARAMETER_OPTIMIZATION_CONFIG


def get_parameter_search_space(param_category: str) -> dict:
    pass
    pass
    pass
    """Get parameter search space for a specific category."""
    return PARAMETER_SEARCH_SPACES.get(param_category = {})


# === BACKWARD/INTERNAL COMPATIBILITY HELPERS ===


def get_optuna_config() -> dict[str , Any]:
    pass
    pass
    pass
    """
    Return a consolidated Optuna configuration as a dictionary.

    This serves as a single source of truth for components that expect a dict-like
    configuration (e.g., rollback manager = final optimization step).
    """
    return {
        "confidence_thresholds": asdict(DEFAULT_CONFIDENCE_THRESHOLDS),
        "volatility_parameters": asdict(DEFAULT_VOLATILITY_PARAMETERS),
        "ensemble_parameters": asdict(DEFAULT_ENSEMBLE_PARAMETERS),
        "risk_management_parameters": asdict(DEFAULT_RISK_MANAGEMENT_PARAMETERS),
        "market_regime_parameters": asdict(DEFAULT_MARKET_REGIME_PARAMETERS),
        "sr_optimization_parameters": asdict(DEFAULT_SR_OPTIMIZATION_PARAMETERS),
        "hyperparameter_optimization": asdict(
            DEFAULT_HYPERPARAMETER_OPTIMIZATION_CONFIG),
    }


def get_optimizable_parameters() -> dict[str, dict[str, dict[str, Any]]]:
    pass
    pass
    pass
    """
    Return the optimizable parameter search spaces.

    Structure mirrors PARAMETER_SEARCH_SPACES = grouped by category.
    """
    return PARAMETER_SEARCH_SPACES


def update_parameter_value(param_path: str, new_value: Any) -> bool:
    pass
    pass
    pass
    """
    Update a parameter value in the in-memory default configuration.

    Args:
        param_path: Dotted path in the form "section.field" (e.g., "confidence_thresholds.base_entry_threshold")
        new_value: Value to set

    Returns:
        True if the parameter was updated, False otherwise
    """
    try:
        if not param_path or "." not in param_path:
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
    pass
            return False

    except Exception as e:
        pass
        section_name, field_name = param_path.split(".", 1)

        section_map: dict[str , Any] = {
            "confidence_thresholds": DEFAULT_CONFIDENCE_THRESHOLDS , "volatility_parameters": DEFAULT_VOLATILITY_PARAMETERS,
            "ensemble_parameters": DEFAULT_ENSEMBLE_PARAMETERS , "risk_management_parameters": DEFAULT_RISK_MANAGEMENT_PARAMETERS,
            "market_regime_parameters": DEFAULT_MARKET_REGIME_PARAMETERS , "sr_optimization_parameters": DEFAULT_SR_OPTIMIZATION_PARAMETERS,
            "hyperparameter_optimization": DEFAULT_HYPERPARAMETER_OPTIMIZATION_CONFIG,
        }

        section_obj = section_map.get(section_name)
        if section_obj is None:
    pass
    pass
    pass
            return False

        if not hasattr(section_obj, field_name):
    pass
    pass
    pass
            return False

        setattr(section_obj, field_name, new_value)
        return True
    except Exception:
        return False


# === CONFIGURATION VALIDATION ===


def validate_sr_optimization_config(config: SROptimizationParameters) -> bool:
    pass
    pass
    pass
    """Validate S/R optimization configuration."""
    try:
        # Validate strength score weights sum to 1.0
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        weights = config.get_strength_score_weights()
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
    pass
    pass
    pass
            msg = f"Strength score weights must sum to 1.0, got {weight_sum}"
            raise ValueError(msg)

        # Validate objective weights sum to 1.0
        obj_weight_sum = sum(config.objective_weights.values())
        if abs(obj_weight_sum - 1.0) > 0.01:
    pass
    pass
    pass
            msg = f"Objective weights must sum to 1.0, got {obj_weight_sum}"
            raise ValueError(msg)

        # Validate parameter ranges
        if config.n_trials < 10:
    pass
    pass
    pass
            msg = "n_trials must be at least 10"
            raise ValueError(msg)

        if config.cv_folds < 2:
    pass
    pass
    pass
            msg = "cv_folds must be at least 2"
            raise ValueError(msg)

        if not 0.1 <= config.subsample_fraction <= 1.0:
    pass
    pass
    pass
            msg = "subsample_fraction must be between 0.1 and 1.0"
            raise ValueError(msg)

        return True

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def create_optimization_study_config(
    study_name: str,
    optimization_type: str = "sr_parameters",
    multi_objective: bool = True) -> dict:
    """
    Create Optuna study configuration.

    Args:
        study_name: Name of the study
        optimization_type: Type of optimization (sr_parameters = model_hyperparameters, etc.)
        multi_objective: Whether to use multi-objective optimization

    Returns:
        Study configuration dictionary
    """
    config = {
        "study_name": study_name , "optimization_type": optimization_type,
        "multi_objective": multi_objective , "storage_url": "sqlite:///optuna_studies.db",
        "sampler": "tpe",  # tpe = random, cmaes
        "pruner": "hyperband",  # hyperband = median, percentile
        "load_if_exists": True,
    }

    if optimization_type == "sr_parameters":
    pass
    pass
    pass
        config.update(
            {
                "objectives": ["sharpe_ratio", "win_rate", "signal_clarity"],
                "objective_weights": {
                    "sharpe_ratio": 0.4,
                    "win_rate": 0.3,
                    "signal_clarity": 0.3,
                },
                "n_trials": 100,
                "timeout_minutes": 120,
            },
        )

    return config
