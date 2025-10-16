"""
Regime Configuration

Configuration for ML-based regime detection including the 15-25 regimes
with percentage weights and detection parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

class RegimeType(Enum):
    """Regime types for classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    RANGE_BOUND = "range_bound"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    TREND_ACCELERATION = "trend_acceleration"
    TREND_DECELERATION = "trend_deceleration"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    CONSOLIDATION = "consolidation"
    BREAKDOWN = "breakdown"
    BREAKOUT_FAILURE = "breakout_failure"
    FALSE_BREAKOUT = "false_breakout"
    TREND_CONTINUATION = "trend_continuation"

@dataclass
class RegimeWeight:
    """Regime weight configuration."""
    regime: RegimeType
    weight: float
    confidence_threshold: float = 0.7
    min_duration: int = 5  # minimum candles
    max_duration: int = 100  # maximum candles

@dataclass
class RegimeConfig:
    """Regime detection configuration."""

    # Regime detection
    enabled_regimes: List[RegimeType] = field(default_factory=lambda: list(RegimeType))
    regime_weights: List[RegimeWeight] = field(default_factory=list)

    # Detection parameters
    lookback_period: int = 50  # candles for regime detection
    confidence_threshold: float = 0.7  # minimum confidence for regime classification
    transition_threshold: float = 0.3  # threshold for regime change detection

    # ML model parameters
    model_type: str = "hmm"  # hmm, lstm, transformer
    model_path: Optional[str] = None
    retrain_interval: int = 24  # hours
    feature_window: int = 20  # candles for feature engineering

    # Regime-specific parameters
    volatility_threshold: float = 0.02  # 2% for volatility regimes
    trend_strength_threshold: float = 0.6  # for trend regimes
    support_resistance_threshold: float = 0.01  # 1% for S/R levels

    # Transition rules
    min_regime_duration: int = 5  # minimum candles in a regime
    max_regime_duration: int = 100  # maximum candles in a regime
    transition_cooldown: int = 3  # candles before allowing transition

    # Regime combinations
    allow_multiple_regimes: bool = True
    max_concurrent_regimes: int = 3
    regime_combination_threshold: float = 0.5

    # Performance tracking
    track_regime_performance: bool = True
    regime_performance_window: int = 100  # candles
    update_regime_weights: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default regime weights if not provided."""
        if not self.regime_weights:
            self._initialize_default_weights()

    def _initialize_default_weights(self):
        """Initialize default regime weights."""
        default_weights = [
            RegimeWeight(RegimeType.TRENDING_UP, 0.15, 0.8),
            RegimeWeight(RegimeType.TRENDING_DOWN, 0.15, 0.8),
            RegimeWeight(RegimeType.SIDEWAYS, 0.10, 0.7),
            RegimeWeight(RegimeType.HIGH_VOLATILITY, 0.08, 0.75),
            RegimeWeight(RegimeType.LOW_VOLATILITY, 0.08, 0.75),
            RegimeWeight(RegimeType.BREAKOUT, 0.12, 0.8),
            RegimeWeight(RegimeType.REVERSAL, 0.10, 0.75),
            RegimeWeight(RegimeType.ACCUMULATION, 0.06, 0.7),
            RegimeWeight(RegimeType.DISTRIBUTION, 0.06, 0.7),
            RegimeWeight(RegimeType.MOMENTUM, 0.10, 0.8),
        ]

        # Add remaining regimes with lower weights
        remaining_regimes = [r for r in RegimeType if r not in [rw.regime for rw in default_weights]]
        for regime in remaining_regimes:
            default_weights.append(RegimeWeight(regime, 0.02, 0.6))

        self.regime_weights = default_weights

    def get_regime_weight(self, regime: RegimeType) -> float:
        """Get weight for a specific regime."""
        for rw in self.regime_weights:
            if rw.regime == regime:
                return rw.weight
        return 0.0

    def get_regime_confidence_threshold(self, regime: RegimeType) -> float:
        """Get confidence threshold for a specific regime."""
        for rw in self.regime_weights:
            if rw.regime == regime:
                return rw.confidence_threshold
        return self.confidence_threshold

    def update_regime_weight(self, regime: RegimeType, new_weight: float):
        """Update weight for a specific regime."""
        for rw in self.regime_weights:
            if rw.regime == regime:
                rw.weight = new_weight
                return
        # Add new regime weight if not found
        self.regime_weights.append(RegimeWeight(regime, new_weight))

    def get_active_regimes(self) -> List[RegimeType]:
        """Get list of active (enabled) regimes."""
        return self.enabled_regimes

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key == "enabled_regimes":
                result[key] = [r.value for r in value]
            elif key == "regime_weights":
                result[key] = [
                    {
                        "regime": rw.regime.value,
                        "weight": rw.weight,
                        "confidence_threshold": rw.confidence_threshold,
                        "min_duration": rw.min_duration,
                        "max_duration": rw.max_duration
                    }
                    for rw in value
                ]
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dict):
                result[key] = value.copy()
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RegimeConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "enabled_regimes" in config_dict:
            config_dict["enabled_regimes"] = [RegimeType(r) for r in config_dict["enabled_regimes"]]

        if "regime_weights" in config_dict:
            weights = []
            for rw_dict in config_dict["regime_weights"]:
                weights.append(RegimeWeight(
                    regime=RegimeType(rw_dict["regime"]),
                    weight=rw_dict["weight"],
                    confidence_threshold=rw_dict.get("confidence_threshold", 0.7),
                    min_duration=rw_dict.get("min_duration", 5),
                    max_duration=rw_dict.get("max_duration", 100)
                ))
            config_dict["regime_weights"] = weights

        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.lookback_period <= 0:
            return False
        if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
            return False
        if self.transition_threshold <= 0 or self.transition_threshold > 1:
            return False
        if self.min_regime_duration <= 0:
            return False
        if self.max_regime_duration <= self.min_regime_duration:
            return False

        # Validate regime weights sum to approximately 1.0
        total_weight = sum(rw.weight for rw in self.regime_weights)
        if abs(total_weight - 1.0) > 0.1:  # Allow 10% tolerance
            return False

        return True
