#!/usr/bin/env python3
"""
Regime-Specific Optimization Configuration

This module extends the existing Optuna configuration to include regime-specific
triple barrier thresholds and TPSL parameters optimization. It provides a comprehensive
configuration system for regime-aware parameter tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from src.config.config_optuna import (
    SROptimizationParameters,
    HyperparameterOptimizationConfig,
    PARAMETER_SEARCH_SPACES
)


class RegimeType(Enum):
    """Enum for different regime types."""

    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS_RANGE = "SIDEWAYS_RANGE"
    HIGH_IMPACT_CANDLE = "HIGH_IMPACT_CANDLE"
    SR_ZONE_ACTION = "SR_ZONE_ACTION"
    VOLATILE_MARKET = "VOLATILE_MARKET"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_MARKET = "TRENDING_MARKET"
    MEAN_REVERSION = "MEAN_REVERSION"


@dataclass
class RegimeSpecificConstraints:
    """Constraints for regime-specific parameter optimization."""

    # Take profit multiplier range
    tp_multiplier_range: List[float] = field(default_factory=lambda: [1.5, 4.0])

    # Stop loss multiplier range
    sl_multiplier_range: List[float] = field(default_factory=lambda: [0.8, 2.0])

    # Position size range
    position_size_range: List[float] = field(default_factory=lambda: [0.05, 0.25])

    # Triple barrier specific constraints
    profit_take_multiplier_range: List[float] = field(default_factory=lambda: [0.01, 0.05])
    stop_loss_multiplier_range: List[float] = field(default_factory=lambda: [0.005, 0.03])
    time_barrier_minutes_range: List[int] = field(default_factory=lambda: [15, 120])
    max_lookahead_range: List[int] = field(default_factory=lambda: [50, 200])

    # Regime-specific multipliers
    volatility_multiplier_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    trend_multiplier_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    volume_multiplier_range: List[float] = field(default_factory=lambda: [0.5, 2.0])

    # TPSL specific parameters
    tp_atr_multiplier_range: List[float] = field(default_factory=lambda: [1.0, 4.0])
    sl_atr_multiplier_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    trailing_stop_range: List[float] = field(default_factory=lambda: [0.0, 0.02])
    break_even_threshold_range: List[float] = field(default_factory=lambda: [0.005, 0.02])


@dataclass
class RegimeSpecificOptimizationConfig:
    """Configuration for regime-specific optimization."""

    # Optimization settings
    enable_regime_optimization: bool = True
    multi_objective: bool = True
    n_trials_per_regime: int = 100
    timeout_minutes_per_regime: int = 60
    cv_folds: int = 5

    # Objectives and weights
    objectives: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "win_rate", "profit_factor", "regime_accuracy"
    ])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.3,
        "win_rate": 0.25,
        "profit_factor": 0.25,
        "regime_accuracy": 0.2
    })

    # Regime-specific constraints
    regime_constraints: Dict[str, RegimeSpecificConstraints] = field(default_factory=lambda: {
        "BULL_TREND": RegimeSpecificConstraints(
            tp_multiplier_range=[2.5, 5.0],
            sl_multiplier_range=[1.2, 2.5],
            position_size_range=[0.10, 0.25],
            profit_take_multiplier_range=[0.02, 0.04],
            stop_loss_multiplier_range=[0.01, 0.02],
        ),
        "BEAR_TREND": RegimeSpecificConstraints(
            tp_multiplier_range=[2.0, 4.5],
            sl_multiplier_range=[1.0, 2.2],
            position_size_range=[0.08, 0.20],
            profit_take_multiplier_range=[0.015, 0.035],
            stop_loss_multiplier_range=[0.008, 0.018],
        ),
        "SIDEWAYS_RANGE": RegimeSpecificConstraints(
            tp_multiplier_range=[1.5, 3.0],
            sl_multiplier_range=[0.8, 1.8],
            position_size_range=[0.06, 0.15],
            profit_take_multiplier_range=[0.01, 0.025],
            stop_loss_multiplier_range=[0.005, 0.015],
        ),
        "HIGH_IMPACT_CANDLE": RegimeSpecificConstraints(
            tp_multiplier_range=[1.8, 3.5],
            sl_multiplier_range=[0.9, 2.0],
            position_size_range=[0.05, 0.12],
            profit_take_multiplier_range=[0.025, 0.045],
            stop_loss_multiplier_range=[0.012, 0.025],
        ),
        "SR_ZONE_ACTION": RegimeSpecificConstraints(
            tp_multiplier_range=[2.0, 4.0],
            sl_multiplier_range=[1.0, 2.2],
            position_size_range=[0.08, 0.18],
            profit_take_multiplier_range=[0.018, 0.035],
            stop_loss_multiplier_range=[0.009, 0.020],
        ),
        "VOLATILE_MARKET": RegimeSpecificConstraints(
            tp_multiplier_range=[2.2, 4.2],
            sl_multiplier_range=[1.1, 2.3],
            position_size_range=[0.06, 0.14],
            profit_take_multiplier_range=[0.025, 0.045],
            stop_loss_multiplier_range=[0.012, 0.025],
        ),
        "LOW_VOLATILITY": RegimeSpecificConstraints(
            tp_multiplier_range=[1.8, 3.2],
            sl_multiplier_range=[0.9, 1.9],
            position_size_range=[0.08, 0.20],
            profit_take_multiplier_range=[0.015, 0.030],
            stop_loss_multiplier_range=[0.007, 0.016],
        ),
        "TRENDING_MARKET": RegimeSpecificConstraints(
            tp_multiplier_range=[2.3, 4.5],
            sl_multiplier_range=[1.1, 2.4],
            position_size_range=[0.09, 0.22],
            profit_take_multiplier_range=[0.020, 0.040],
            stop_loss_multiplier_range=[0.010, 0.020],
        ),
        "MEAN_REVERSION": RegimeSpecificConstraints(
            tp_multiplier_range=[1.6, 3.2],
            sl_multiplier_range=[0.8, 1.8],
            position_size_range=[0.07, 0.16],
            profit_take_multiplier_range=[0.012, 0.028],
            stop_loss_multiplier_range=[0.006, 0.015],
        ),
    })

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_delta: float = 0.001

    # Pruning settings
    enable_pruning: bool = True
    pruning_method: str = "hyperband"

    # Statistical testing
    enable_statistical_testing: bool = True
    confidence_level: float = 0.95
    min_sample_size: int = 50

    # Regime mapping
    regime_id_to_name: Dict[int, str] = field(default_factory=dict)
    regime_name_to_id: Dict[str, int] = field(default_factory=dict)

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.55
    min_profit_factor: float = 1.3
    max_drawdown_threshold: float = -0.15

    # Optimization storage
    storage_url: str = "sqlite:///regime_triple_barrier_optuna_studies.db"
    study_name_prefix: str = "regime_triple_barrier_optimization"

    def get_regime_constraints(self, regime_name: str) -> Optional[RegimeSpecificConstraints]:
        """Get constraints for a specific regime."""
        return self.regime_constraints.get(regime_name)

    def add_regime_constraints(self, regime_name: str, constraints: RegimeSpecificConstraints):
        """Add constraints for a new regime."""
        self.regime_constraints[regime_name] = constraints
        self._initialize_regime_mapping()

    def update_regime_constraints(self, regime_name: str, constraints: RegimeSpecificConstraints):
        """Update constraints for an existing regime."""
        if regime_name in self.regime_constraints:
            self.regime_constraints[regime_name] = constraints
        else:
            self.add_regime_constraints(regime_name, constraints)


# Extended parameter search spaces for regime-specific optimization
REGIME_SPECIFIC_PARAMETER_SEARCH_SPACES = {
    "regime_triple_barrier_parameters": {
        # Base triple barrier parameters
        "profit_take_multiplier": {"min": 0.01, "max": 0.05, "type": "float", "log": True},
        "stop_loss_multiplier": {"min": 0.005, "max": 0.03, "type": "float", "log": True},
        "time_barrier_minutes": {"min": 15, "max": 120, "type": "int"},
        "max_lookahead": {"min": 50, "max": 200, "type": "int"},

        # Regime-specific multipliers
        "regime_volatility_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},
        "regime_trend_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},
        "regime_volume_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},

        # TPSL parameters
        "tp_multiplier": {"min": 1.5, "max": 4.0, "type": "float"},
        "sl_multiplier": {"min": 0.8, "max": 2.0, "type": "float"},
        "position_size": {"min": 0.05, "max": 0.25, "type": "float"},

        # Advanced TPSL parameters
        "tp_atr_multiplier": {"min": 1.0, "max": 4.0, "type": "float"},
        "sl_atr_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},
        "trailing_stop": {"min": 0.0, "max": 0.02, "type": "float"},
        "break_even_threshold": {"min": 0.005, "max": 0.02, "type": "float"},
    },

    "regime_performance_metrics": {
        "sharpe_ratio": {"min": -2.0, "max": 3.0, "type": "float"},
        "win_rate": {"min": 0.0, "max": 1.0, "type": "float"},
        "profit_factor": {"min": 0.5, "max": 3.0, "type": "float"},
        "total_return": {"min": -0.5, "max": 1.0, "type": "float"},
        "max_drawdown": {"min": -0.5, "max": 0.0, "type": "float"},
        "sortino_ratio": {"min": -2.0, "max": 3.0, "type": "float"},
        "calmar_ratio": {"min": -2.0, "max": 5.0, "type": "float"},
        "regime_accuracy": {"min": 0.0, "max": 1.0, "type": "float"},
        "regime_precision": {"min": 0.0, "max": 1.0, "type": "float"},
        "regime_recall": {"min": 0.0, "max": 1.0, "type": "float"},
        "regime_f1": {"min": 0.0, "max": 1.0, "type": "float"},
    }
}


# Default configuration instances
DEFAULT_REGIME_SPECIFIC_OPTIMIZATION_CONFIG = RegimeSpecificOptimizationConfig()






# Utility functions for regime-specific optimization

def validate_regime_optimization_config(config: RegimeSpecificOptimizationConfig) -> bool:
    """Validate regime-specific optimization configuration."""

    try:
        # Check objectives
        if not config.objectives:
            print("❌ No objectives specified")
            return False

        # Check objective weights
        weight_sum = sum(config.objective_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            print(f"❌ Objective weights must sum to 1.0, got {weight_sum}")
            return False

        # Check regime constraints
        if not config.regime_constraints:
            print("❌ No regime constraints specified")
            return False

        # Check parameter ranges
        for regime_name, constraints in config.regime_constraints.items():
            if constraints.tp_multiplier_range[0] >= constraints.tp_multiplier_range[1]:
                print(f"❌ Invalid TP multiplier range for {regime_name}")
                return False

            if constraints.sl_multiplier_range[0] >= constraints.sl_multiplier_range[1]:
                print(f"❌ Invalid SL multiplier range for {regime_name}")
                return False

            if constraints.position_size_range[0] >= constraints.position_size_range[1]:
                print(f"❌ Invalid position size range for {regime_name}")
                return False

        # Check optimization settings
        if config.n_trials_per_regime < 10:
            print("❌ n_trials_per_regime must be at least 10")
            return False

        if config.cv_folds < 2:
            print("❌ cv_folds must be at least 2")
            return False

        if config.min_sample_size < 10:
            print("❌ min_sample_size must be at least 10")
            return False

        print("✅ Regime-specific optimization configuration is valid")
        return True

    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        return False