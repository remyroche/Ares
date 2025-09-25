"""
TAS Regime Configuration

Simple configuration class for TAS regime detection.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TASRegimeConfig:
    """Configuration for TAS regime detection."""
    
    n_regimes: int = 5
    primary_timeframe: str = "1m"
    tree_depth: int = 10
    n_estimators: int = 100
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[str] = None
    enable_statistical_methods: bool = True
    enable_bootstrap_analysis: bool = True
    bootstrap_iterations: int = 100
    enable_clvsa_enhancement: bool = True
    enable_regime_adaptation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_multi_scale_analysis: bool = True
    enable_hardware_optimization: bool = True
    enable_matrix_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_economic_evaluation: bool = True
    economic_significance_threshold: float = 0.05
    trading_viability_threshold: float = 0.6
    enable_meta_learning: bool = True
    adaptation_rate: float = 0.01
    memory_size: int = 1000