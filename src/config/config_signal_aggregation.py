"""
Signal Aggregation Configuration with Step17 Optimization Support
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SignalAggregationConfig:
    """Configuration for signal aggregation with step17 optimization."""
    
    # Source weights for different signal sources
    analyst_weight: float = 0.5
    tactician_weight: float = 0.5
    scenario_weight: float = 0.3
    sr_breakout_weight: float = 0.2
    regime_weight: float = 0.4
    
    # Aggregation method
    use_multiplicative: bool = True
    
    # Conflict penalty
    conflict_penalty_factor: float = 0.5
    
    # Minimum weights to prevent signal suppression
    min_source_weight: float = 0.1
    
    # Confidence thresholds
    min_signal_confidence: float = 0.3
    min_aggregated_confidence: float = 0.5
    
    # Direction alignment bonuses
    regime_alignment_bonus: float = 0.2
    multi_signal_alignment_bonus: float = 0.1
    
    def __post_init__(self):
        # Normalize weights to sum to a reasonable value
        total_weight = (
            self.analyst_weight + 
            self.tactician_weight + 
            self.scenario_weight + 
            self.sr_breakout_weight + 
            self.regime_weight
        )
        if total_weight > 2.0:
            # Normalize if weights are too high
            scale = 2.0 / total_weight
            self.analyst_weight *= scale
            self.tactician_weight *= scale
            self.scenario_weight *= scale
            self.sr_breakout_weight *= scale
            self.regime_weight *= scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "analyst_weight": self.analyst_weight,
            "tactician_weight": self.tactician_weight,
            "scenario_weight": self.scenario_weight,
            "sr_breakout_weight": self.sr_breakout_weight,
            "regime_weight": self.regime_weight,
            "use_multiplicative": self.use_multiplicative,
            "conflict_penalty_factor": self.conflict_penalty_factor,
            "min_source_weight": self.min_source_weight,
            "min_signal_confidence": self.min_signal_confidence,
            "min_aggregated_confidence": self.min_aggregated_confidence,
            "regime_alignment_bonus": self.regime_alignment_bonus,
            "multi_signal_alignment_bonus": self.multi_signal_alignment_bonus
        }


def get_step17_search_space() -> Dict[str, Dict[str, Any]]:
    """Get search space for step17 optimization."""
    return {
        "analyst_weight": {"min": 0.2, "max": 0.8, "type": "float"},
        "tactician_weight": {"min": 0.2, "max": 0.8, "type": "float"},
        "scenario_weight": {"min": 0.1, "max": 0.5, "type": "float"},
        "sr_breakout_weight": {"min": 0.1, "max": 0.4, "type": "float"},
        "regime_weight": {"min": 0.2, "max": 0.6, "type": "float"},
        "conflict_penalty_factor": {"min": 0.3, "max": 0.7, "type": "float"},
        "min_source_weight": {"min": 0.05, "max": 0.2, "type": "float"},
        "min_signal_confidence": {"min": 0.2, "max": 0.5, "type": "float"},
        "min_aggregated_confidence": {"min": 0.4, "max": 0.7, "type": "float"},
        "regime_alignment_bonus": {"min": 0.1, "max": 0.3, "type": "float"},
        "multi_signal_alignment_bonus": {"min": 0.05, "max": 0.2, "type": "float"},
        "use_multiplicative": {"choices": [True, False], "type": "categorical"}
    }


def create_default_config() -> SignalAggregationConfig:
    """Create default signal aggregation configuration."""
    return SignalAggregationConfig()


def update_from_step17(
    config: SignalAggregationConfig, 
    step17_results: Dict[str, Any]
) -> SignalAggregationConfig:
    """
    Update configuration with step17 optimization results.
    
    Args:
        config: Current configuration
        step17_results: Optimization results from step17
        
    Returns:
        Updated configuration
    """
    if "signal_aggregation" in step17_results:
        optimized = step17_results["signal_aggregation"]
        
        # Update weights
        config.analyst_weight = optimized.get("analyst_weight", config.analyst_weight)
        config.tactician_weight = optimized.get("tactician_weight", config.tactician_weight)
        config.scenario_weight = optimized.get("scenario_weight", config.scenario_weight)
        config.sr_breakout_weight = optimized.get("sr_breakout_weight", config.sr_breakout_weight)
        config.regime_weight = optimized.get("regime_weight", config.regime_weight)
        
        # Update other parameters
        config.use_multiplicative = optimized.get("use_multiplicative", config.use_multiplicative)
        config.conflict_penalty_factor = optimized.get("conflict_penalty_factor", config.conflict_penalty_factor)
        config.min_source_weight = optimized.get("min_source_weight", config.min_source_weight)
        config.min_signal_confidence = optimized.get("min_signal_confidence", config.min_signal_confidence)
        config.min_aggregated_confidence = optimized.get("min_aggregated_confidence", config.min_aggregated_confidence)
        config.regime_alignment_bonus = optimized.get("regime_alignment_bonus", config.regime_alignment_bonus)
        config.multi_signal_alignment_bonus = optimized.get("multi_signal_alignment_bonus", config.multi_signal_alignment_bonus)
        
        # Re-normalize after update
        config.__post_init__()
    
    return config