"""
Unified Regime Detection Configuration

This module provides a unified configuration system that combines the best aspects
of both TAS (Tree Architecture Search) and NAS (Neural Architecture Search) regime
detection systems with enhanced economic significance and trading viability evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ._validation import (
    ConfigValidationError,
    ValidationIssue,
    build_report,
    ensure_between,
    ensure_min_less_than_max,
    ensure_non_empty_string,
    ensure_positive,
    ensure_probability,
    normalize_weights,
)

logger = logging.getLogger(__name__)

class RegimeDetectionMethod(Enum):
    """Unified regime detection methods."""
    TAS_ONLY = "tas_only"
    NAS_ONLY = "nas_only"
    HYBRID = "hybrid"
    ADAPTIVE_SELECTION = "adaptive_selection"

class OptimizationStrategy(Enum):
    """Optimization strategies for regime detection."""
    PERFORMANCE_FIRST = "performance_first"
    ACCURACY_FIRST = "accuracy_first"
    BALANCED = "balanced"
    ECONOMIC_FOCUSED = "economic_focused"

class EconomicEvaluationMode(Enum):
    """Economic evaluation modes."""
    BASIC = "basic"
    ADVANCED = "advanced"
    POSITION_AWARE = "position_aware"
    REAL_TIME = "real_time"

@dataclass
class EconomicEvaluationConfig:
    """Configuration for economic evaluation."""
    enable_position_aware_analysis: bool = True
    enable_risk_adjusted_metrics: bool = True
    enable_trading_cost_analysis: bool = True
    enable_liquidity_analysis: bool = True
    position_size_threshold: float = 0.01
    max_position_size: float = 0.1
    risk_free_rate: float = 0.02
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0

    def validate(self) -> None:
        """Validate economic evaluation parameters."""

        ensure_probability("position_size_threshold", self.position_size_threshold)
        ensure_probability("max_position_size", self.max_position_size)
        ensure_min_less_than_max(
            "position_size_threshold",
            self.position_size_threshold,
            "max_position_size",
            self.max_position_size,
        )
        ensure_between("risk_free_rate", self.risk_free_rate, minimum=-1.0, maximum=1.0)
        ensure_positive("transaction_cost_bps", self.transaction_cost_bps, allow_zero=True)
        ensure_positive("slippage_bps", self.slippage_bps, allow_zero=True)

@dataclass
class TradingConfig:
    """Configuration for trading viability evaluation."""
    enable_trading_signals: bool = True
    enable_risk_management: bool = True
    enable_position_sizing: bool = True
    min_hold_time: int = 5  # minutes
    max_hold_time: int = 1440  # minutes (24 hours)
    stop_loss_threshold: float = 0.02
    take_profit_threshold: float = 0.04
    max_drawdown_threshold: float = 0.1
    viability_threshold: float = 0.7

    def validate(self) -> None:
        """Validate trading configuration values."""

        ensure_positive("min_hold_time", self.min_hold_time)
        ensure_positive("max_hold_time", self.max_hold_time)
        ensure_min_less_than_max("min_hold_time", self.min_hold_time, "max_hold_time", self.max_hold_time)
        ensure_probability("stop_loss_threshold", self.stop_loss_threshold)
        ensure_probability("take_profit_threshold", self.take_profit_threshold)
        ensure_probability("max_drawdown_threshold", self.max_drawdown_threshold)
        ensure_probability("viability_threshold", self.viability_threshold)

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning capabilities."""
    enable_meta_learning: bool = True
    enable_architecture_adaptation: bool = True
    enable_hyperparameter_adaptation: bool = True
    adaptation_frequency: int = 100  # epochs
    learning_rate_adaptation: bool = True
    architecture_mutation_rate: float = 0.1
    hyperparameter_mutation_rate: float = 0.05

    def validate(self) -> None:
        """Validate meta-learning configuration values."""

        ensure_positive("adaptation_frequency", self.adaptation_frequency)
        ensure_probability("architecture_mutation_rate", self.architecture_mutation_rate)
        ensure_probability("hyperparameter_mutation_rate", self.hyperparameter_mutation_rate)

@dataclass
class UnifiedRegimeConfig:
    """Unified configuration for regime detection systems."""
    
    # Core detection parameters
    detection_method: RegimeDetectionMethod = RegimeDetectionMethod.HYBRID
    n_regimes: int = 5
    primary_timeframe: str = "1h"
    micro_timeframe: str = "15m"
    macro_timeframe: str = "4h"
    
    # Sample constraints
    min_regime_samples: int = 100
    max_regime_samples: int = 10000
    
    # Economic evaluation configuration
    economic_evaluation: EconomicEvaluationMode = EconomicEvaluationMode.POSITION_AWARE
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    risk_adjusted_return_threshold: float = 0.1
    
    # Economic evaluation parameters
    economic_params: Dict[str, Any] = field(default_factory=lambda: {
        'enable_position_aware_analysis': True,
        'enable_risk_adjusted_metrics': True,
        'enable_trading_cost_analysis': True,
        'enable_liquidity_analysis': True,
        'position_size_threshold': 0.01,
        'max_position_size': 0.1,
        'risk_free_rate': 0.02,
        'transaction_cost_bps': 5.0,
        'slippage_bps': 2.0
    })
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    
    # Performance thresholds
    min_regime_stability: float = 0.6
    min_transition_confidence: float = 0.7
    max_execution_time: float = 300.0  # seconds
    target_accuracy: float = 0.85
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 512
    gpu_memory_fraction: float = 0.8
    
    # Data quality requirements
    min_data_quality: float = 0.8
    max_missing_data_ratio: float = 0.05
    enable_data_validation: bool = True
    data_quality_threshold: float = 0.8
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_visualization: bool = True
    results_directory: str = "unified_regime_results"
    
    # Component configurations
    economic_config: EconomicEvaluationConfig = field(default_factory=EconomicEvaluationConfig)
    trading_config: TradingConfig = field(default_factory=TradingConfig)
    meta_learning_config: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    
    # Hybrid system weights
    tas_weight: float = 0.6
    nas_weight: float = 0.4
    adaptive_weighting: bool = True
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate configuration parameters."""

        issues: List[ValidationIssue] = []

        def capture(field: str, func) -> None:
            try:
                func()
            except ConfigValidationError as exc:  # pragma: no cover - defensive aggregation
                issues.append(ValidationIssue(field=field, message=str(exc)))

        capture("n_regimes", lambda: ensure_between("n_regimes", self.n_regimes, minimum=2, maximum=20))
        capture("min_regime_samples", lambda: ensure_positive("min_regime_samples", self.min_regime_samples))
        capture("max_regime_samples", lambda: ensure_positive("max_regime_samples", self.max_regime_samples))
        capture(
            "sample_range",
            lambda: ensure_min_less_than_max(
                "min_regime_samples",
                float(self.min_regime_samples),
                "max_regime_samples",
                float(self.max_regime_samples),
            ),
        )
        capture("primary_timeframe", lambda: ensure_non_empty_string("primary_timeframe", self.primary_timeframe))
        capture("micro_timeframe", lambda: ensure_non_empty_string("micro_timeframe", self.micro_timeframe))
        capture("macro_timeframe", lambda: ensure_non_empty_string("macro_timeframe", self.macro_timeframe))

        for threshold_name in (
            "economic_significance_threshold",
            "trading_viability_threshold",
            "min_regime_stability",
            "min_transition_confidence",
            "target_accuracy",
            "min_data_quality",
            "data_quality_threshold",
        ):
            capture(
                threshold_name,
                lambda name=threshold_name: ensure_probability(name, getattr(self, name)),
            )

        capture("max_missing_data_ratio", lambda: ensure_probability("max_missing_data_ratio", self.max_missing_data_ratio))
        capture("max_execution_time", lambda: ensure_positive("max_execution_time", self.max_execution_time))

        weight_map = {"tas_weight": self.tas_weight, "nas_weight": self.nas_weight}
        capture("weight_normalization", lambda: normalize_weights(weight_map))
        if "tas_weight" in weight_map:
            self.tas_weight = weight_map["tas_weight"]
            self.nas_weight = weight_map["nas_weight"]

        required_params = {
            "enable_position_aware_analysis",
            "enable_risk_adjusted_metrics",
            "enable_trading_cost_analysis",
            "enable_liquidity_analysis",
            "position_size_threshold",
            "max_position_size",
            "risk_free_rate",
            "transaction_cost_bps",
            "slippage_bps",
        }
        missing_params = required_params - set(self.economic_params)
        if missing_params:
            issues.append(
                ValidationIssue(
                    field="economic_params",
                    message=f"Missing required keys: {sorted(missing_params)}",
                )
            )
        else:
            capture(
                "economic_params.position_size_threshold",
                lambda: ensure_probability(
                    "economic_params.position_size_threshold",
                    float(self.economic_params["position_size_threshold"]),
                ),
            )
            capture(
                "economic_params.max_position_size",
                lambda: ensure_probability(
                    "economic_params.max_position_size",
                    float(self.economic_params["max_position_size"]),
                ),
            )
            capture(
                "economic_params.range",
                lambda: ensure_min_less_than_max(
                    "economic_params.position_size_threshold",
                    float(self.economic_params["position_size_threshold"]),
                    "economic_params.max_position_size",
                    float(self.economic_params["max_position_size"]),
                ),
            )
            capture(
                "economic_params.risk_free_rate",
                lambda: ensure_between(
                    "economic_params.risk_free_rate",
                    float(self.economic_params["risk_free_rate"]),
                    minimum=-1.0,
                    maximum=1.0,
                ),
            )
            capture(
                "economic_params.transaction_cost_bps",
                lambda: ensure_positive(
                    "economic_params.transaction_cost_bps",
                    float(self.economic_params["transaction_cost_bps"]),
                    allow_zero=True,
                ),
            )
            capture(
                "economic_params.slippage_bps",
                lambda: ensure_positive(
                    "economic_params.slippage_bps",
                    float(self.economic_params["slippage_bps"]),
                    allow_zero=True,
                ),
            )

        for name, validator in (
            ("economic_config", self.economic_config.validate),
            ("trading_config", self.trading_config.validate),
            ("meta_learning_config", self.meta_learning_config.validate),
        ):
            try:
                validator()
            except ConfigValidationError as exc:  # pragma: no cover - defensive aggregation
                issues.append(ValidationIssue(field=name, message=str(exc)))

        if issues:
            build_report(*issues)
            raise ConfigValidationError(
                "UnifiedRegimeConfig validation failed; inspect log output for details."
            )

        logger.info("✅ Unified regime configuration validation passed")
    
    def get_tas_config(self) -> Dict[str, Any]:
        """Get TAS-specific configuration."""
        return {
            'n_regimes': self.n_regimes,
            'primary_timeframe': self.primary_timeframe,
            'min_regime_samples': self.min_regime_samples,
            'max_regime_samples': self.max_regime_samples,
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'max_execution_time': self.max_execution_time,
            'enable_hardware_optimization': self.enable_hardware_optimization,
            'optimization_strategy': self.optimization_strategy.value,
            'economic_evaluation': self.economic_evaluation.value,
            'economic_params': self.economic_params,
            'trading_config': self.trading_config.__dict__,
            'meta_learning_config': self.meta_learning_config.__dict__
        }
    
    def get_nas_config(self) -> Dict[str, Any]:
        """Get NAS-specific configuration."""
        return {
            'n_regimes': self.n_regimes,
            'primary_timeframe': self.primary_timeframe,
            'min_regime_samples': self.min_regime_samples,
            'max_regime_samples': self.max_regime_samples,
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'max_execution_time': self.max_execution_time,
            'enable_hardware_optimization': self.enable_hardware_optimization,
            'optimization_strategy': self.optimization_strategy.value,
            'economic_evaluation': self.economic_evaluation.value,
            'economic_params': self.economic_params,
            'trading_config': self.trading_config.__dict__,
            'meta_learning_config': self.meta_learning_config.__dict__
        }
    
    def get_hybrid_config(self) -> Dict[str, Any]:
        """Get hybrid configuration."""
        base_config = self.get_tas_config()
        base_config.update({
            'tas_weight': self.tas_weight,
            'nas_weight': self.nas_weight,
            'adaptive_weighting': self.adaptive_weighting,
            'detection_method': self.detection_method.value
        })
        return base_config
    
    def is_valid_for_trading(self) -> bool:
        """Check if configuration is valid for trading."""
        return (
            self.economic_significance_threshold >= 0.7 and
            self.trading_viability_threshold >= 0.6 and
            self.max_execution_time <= 300 and
            self.min_regime_stability >= 0.6
        )
    
    @classmethod
    def create_tas_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for TAS-only detection."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.TAS_ONLY
        config.optimization_strategy = OptimizationStrategy.ACCURACY_FIRST
        config.economic_evaluation = EconomicEvaluationMode.POSITION_AWARE
        config.tas_weight = 1.0
        config.nas_weight = 0.0
        config.adaptive_weighting = False
        return config
    
    @classmethod
    def create_nas_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for NAS-only detection."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.NAS_ONLY
        config.optimization_strategy = OptimizationStrategy.PERFORMANCE_FIRST
        config.economic_evaluation = EconomicEvaluationMode.ADVANCED
        config.tas_weight = 0.0
        config.nas_weight = 1.0
        config.adaptive_weighting = False
        return config
    
    @classmethod
    def create_hybrid_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for hybrid detection."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID
        config.optimization_strategy = OptimizationStrategy.BALANCED
        config.economic_evaluation = EconomicEvaluationMode.POSITION_AWARE
        config.tas_weight = 0.6
        config.nas_weight = 0.4
        config.adaptive_weighting = True
        return config
    
    @classmethod
    def create_trading_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for short-term trading."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID
        config.optimization_strategy = OptimizationStrategy.PERFORMANCE_FIRST
        config.economic_evaluation = EconomicEvaluationMode.REAL_TIME
        config.primary_timeframe = "15m"
        config.micro_timeframe = "5m"
        config.macro_timeframe = "1h"
        config.economic_significance_threshold = 0.8
        config.trading_viability_threshold = 0.7
        config.max_execution_time = 60
        config.trading_config.viability_threshold = 0.6
        config.max_execution_time = 120
        return config
    
    @classmethod
    def create_research_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for research and experimentation."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID
        config.optimization_strategy = OptimizationStrategy.ACCURACY_FIRST
        config.economic_evaluation = EconomicEvaluationMode.ADVANCED
        config.enable_visualization = True
        config.enable_detailed_logging = True
        config.max_execution_time = 600
        return config
    
    @classmethod
    def create_production_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for production deployment."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID
        config.optimization_strategy = OptimizationStrategy.PERFORMANCE_FIRST
        config.economic_evaluation = EconomicEvaluationMode.POSITION_AWARE
        config.enable_hardware_optimization = True
        config.enable_gpu_acceleration = True
        config.enable_memory_optimization = True
        config.enable_caching = True
        config.max_execution_time = 300
        config.economic_significance_threshold = 0.75
        config.trading_viability_threshold = 0.65
        config.min_regime_stability = 0.7
        config.enable_early_stopping = True
        config.early_stopping_patience = 5
        return config
    
    def get_economic_config(self) -> Dict[str, Any]:
        """Get economic evaluation configuration."""
        return self.economic_config.__dict__
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.trading_config.__dict__
    
    def get_meta_learning_config(self) -> Dict[str, Any]:
        """Get meta-learning configuration."""
        return self.meta_learning_config.__dict__