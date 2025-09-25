"""
Perfect NAS Configuration

Configuration class for NAS regime detection.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PerfectNASConfig:
    """Configuration for NAS regime detection."""
    
    n_regimes: int = 5
    primary_timeframe: str = "1m"
    enable_neural_odes: bool = True
    enable_vision_transformers: bool = True
    enable_state_space_models: bool = True
    enable_meta_learning: bool = True
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    accuracy_threshold: float = 0.8
    economic_significance_threshold: float = 0.05
    trading_viability_threshold: float = 0.6
    regime_stability_threshold: float = 0.7
    transition_accuracy_threshold: float = 0.8
    max_execution_time: int = 3600
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10