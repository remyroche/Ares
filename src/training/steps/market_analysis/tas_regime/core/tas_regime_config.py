"""
Tree-Driven Advanced Statistics (TAS) Regime Configuration

This module provides configuration for the tree-driven regime detection system
that combines advanced statistical methods with tree-based learning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

class TASArchitectureType(Enum):
    """Types of TAS architectures available."""
    HYBRID_TREE = "hybrid_tree"
    STATISTICAL_TREE = "statistical_tree"
    ADAPTIVE_TREE = "adaptive_tree"
    ENSEMBLE_TREE = "ensemble_tree"

class TASOptimizationLevel(Enum):
    """Optimization levels for TAS regime detection."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

@dataclass
class TASRegimeConfig:
    """Configuration for TAS Regime Detection System."""

    # Core parameters
    n_regimes: int = 8
    primary_timeframe: str = "15m"
    min_regime_samples: int = 50
    max_regime_samples: int = 10000

    # Tree architecture configuration
    primary_architecture: TASArchitectureType = TASArchitectureType.HYBRID_TREE
    tree_depth: int = 6
    n_estimators: int = 1000
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: Union[str, float] = 'sqrt'

    # Statistical methods
    enable_statistical_methods: bool = True
    statistical_significance_level: float = 0.05
    enable_bootstrap_analysis: bool = True
    bootstrap_iterations: int = 1000

    # Advanced features
    enable_patchtst_enhancement: bool = True
    enable_regime_adaptation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_multi_scale_analysis: bool = True

    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_matrix_optimization: bool = True
    enable_memory_optimization: bool = True
    optimization_level: TASOptimizationLevel = TASOptimizationLevel.MAXIMUM
    enable_streaming_regime_detection: bool = False
    streaming_chunk_size: int = 50000
    max_parallel_timeframes: Optional[int] = None

    # Economic evaluation
    enable_economic_evaluation: bool = True
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    risk_adjusted_return_threshold: float = 0.1

    # Meta-learning
    enable_meta_learning: bool = True
    adaptation_rate: float = 0.1
    memory_size: int = 1000

    # Regime discovery parameters
    regime_discovery_method: str = "hybrid"
    stability_threshold: float = 0.8
    transition_detection_sensitivity: float = 0.7

    # Performance parameters
    max_execution_time: float = 300.0  # seconds
    target_accuracy: float = 0.85
    enable_early_stopping: bool = True

    # Data parameters
    feature_importance_threshold: float = 0.1
    correlation_threshold: float = 0.8
    outlier_detection_enabled: bool = True

    # Caching configuration
    enable_result_caching: bool = True
    cache_base_directory: str = "tas_regime_cache"
    cache_namespace: str = "regime_results"
    cache_ttl_hours: int = 6
    cache_max_entries: int = 64
    cache_eviction_policy: str = "lru"

    @classmethod
    def create_short_term_trading_config(cls) -> 'TASRegimeConfig':
        """Create configuration optimized for short-term trading (5-30m)."""
        return cls(
            n_regimes=6,
            primary_timeframe="15m",
            tree_depth=8,
            n_estimators=1500,
            enable_patchtst_enhancement=True,
            enable_economic_evaluation=True,
            enable_meta_learning=True,
            optimization_level=TASOptimizationLevel.MAXIMUM,
            max_execution_time=120.0
        )

    @classmethod
    def create_research_config(cls) -> 'TASRegimeConfig':
        """Create configuration for research with maximum capabilities."""
        return cls(
            n_regimes=12,
            primary_timeframe="15m",
            tree_depth=10,
            n_estimators=2000,
            enable_patchtst_enhancement=True,
            enable_statistical_methods=True,
            enable_bootstrap_analysis=True,
            enable_meta_learning=True,
            optimization_level=TASOptimizationLevel.MAXIMUM,
            max_execution_time=600.0
        )

    @classmethod
    def create_production_config(cls) -> 'TASRegimeConfig':
        """Create configuration for production deployment."""
        return cls(
            n_regimes=8,
            primary_timeframe="15m",
            tree_depth=6,
            n_estimators=1000,
            enable_patchtst_enhancement=True,
            enable_economic_evaluation=True,
            optimization_level=TASOptimizationLevel.ADVANCED,
            max_execution_time=60.0,
            enable_early_stopping=True
        )

    # Multi-timeframe settings
    trading_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    regime_detection_timeframe: str = "1h"
    enable_multi_timeframe_training: bool = True

    # NAS integration settings
    enable_nas_integration: bool = True
    nas_search_strategy: str = "evolutionary"
    nas_population_size: int = 50
    nas_generations: int = 100

    # Multi-objective optimization
    enable_multi_objective: bool = True
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.3,
        "economic_significance": 0.25,
        "trading_viability": 0.25,
        "computational_efficiency": 0.1,
        "architecture_complexity": 0.1
    })

    # System identification
    system_name: str = "Tree Architecture Search (TAS) Regime System"
    version: str = "1.0.0"

    # Execution settings
    max_execution_time: int = 300  # seconds
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10

    # Logging and monitoring
    log_level: str = "INFO"
    enable_profiling: bool = True
    enable_visualization: bool = True
    save_results: bool = True
    results_directory: str = "tas_regime_results"

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if self.n_regimes < 2 or self.n_regimes > 20:
            raise ValueError("n_regimes must be between 2 and 20")

        if self.tree_depth < 3 or self.tree_depth > 15:
            raise ValueError("tree_depth must be between 3 and 15")

        if self.n_estimators < 100 or self.n_estimators > 5000:
            raise ValueError("n_estimators must be between 100 and 5000")

        # Validate timeframes
        if self.regime_detection_timeframe not in self.trading_timeframes:
            raise ValueError("regime_detection_timeframe must be in trading_timeframes")

        # Validate objective weights
        total_weight = sum(self.objective_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            for obj in self.objective_weights:
                self.objective_weights[obj] /= total_weight

        if self.cache_ttl_hours <= 0:
            raise ValueError("cache_ttl_hours must be positive")
        if self.cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be positive")
        if self.cache_eviction_policy not in {"lru", "lfu", "fifo"}:
            raise ValueError("cache_eviction_policy must be one of 'lru', 'lfu', or 'fifo'")

        return True
