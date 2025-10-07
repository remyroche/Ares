"""
Configuration for Data-Driven Feature Selection System

This module defines the configuration classes for the data-driven feature selection
system, including parameters for both phases of the gating process, budget constraints,
and final model selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np


class FeatureFamily(Enum):
    """Feature families for coverage requirements."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    ANCHOR_TOD = "anchor_tod"
    LIQUIDITY = "liquidity"
    MEAN_REVERSION = "mean_reversion"
    MODEL_SYNERGY = "model_synergy"


class ProbeType(Enum):
    """Types of probes for feature evaluation."""
    COARSE_GRID = "coarse_grid"
    SHORT_WINDOW = "short_window"
    REDUCED_HORIZON = "reduced_horizon"
    SUBSET_DAYS = "subset_days"
    COARSER_BAR = "coarser_bar"


@dataclass
class BudgetConfig:
    """Budget configuration for feature selection."""
    # Compute budget constraints
    feature_compute_p99_budget_ms: float = 50.0  # P99 latency budget in ms
    max_features_pre_selection: int = 120  # Maximum features before final selection
    max_interactions: int = 15  # Maximum interaction features
    max_final_features: int = 60  # Maximum final features (target 45)
    
    # Cost penalties
    lambda_cost: float = 0.1  # Cost penalty weight
    lambda_uncertainty: float = 0.2  # Uncertainty penalty weight
    lambda_staleness: float = 0.05  # Staleness penalty weight
    
    # Coverage requirements
    min_families_covered: int = 4  # Minimum number of families to cover
    required_families: List[FeatureFamily] = field(default_factory=lambda: [
        FeatureFamily.MOMENTUM,
        FeatureFamily.VOLATILITY,
        FeatureFamily.ANCHOR_TOD,
        FeatureFamily.LIQUIDITY
    ])
    
    # Diversification
    correlation_threshold: float = 0.9  # Correlation threshold for diversification
    diversification_penalty: float = 0.15  # Penalty for correlated features


@dataclass
class Phase1Config:
    """Configuration for Phase 1: Cheap Probes."""
    # Data sampling
    probe_days: int = 20  # Number of days for cheap probes
    coarser_bar_multiplier: int = 3  # Use 15m bars if trading 5m (3x cost reduction)
    subset_ratio: float = 0.3  # Use 30% of available data
    
    # Lookback grids (coarse)
    momentum_lookbacks: List[int] = field(default_factory=lambda: [5, 12])
    volatility_lookbacks: List[int] = field(default_factory=lambda: [6, 12])
    rsi_lookbacks: List[int] = field(default_factory=lambda: [7, 14])
    vwap_lookbacks: List[int] = field(default_factory=lambda: [10, 20])
    
    # Transform settings
    default_transform: str = "ew_z"  # Default transformation
    single_horizon: int = 1  # Only use h=1 for probes
    
    # Evaluation settings
    min_pass_rate: float = 0.6  # Minimum pass rate across folds
    min_utility_threshold: float = 0.0  # Minimum utility score
    correlation_threshold: float = 0.9  # Redundancy threshold
    
    # Contextual baselines
    include_context_baselines: bool = True
    context_baselines: List[str] = field(default_factory=lambda: [
        "index_return", "session_dummy", "open_close"
    ])
    
    # Purged OOS settings
    embargo_periods: int = 1  # Embargo periods (largest proxy lookback)
    block_bootstrap_blocks: int = 5  # Number of blocks for bootstrap


@dataclass
class Phase2Config:
    """Configuration for Phase 2: Rich Probes."""
    # Bayesian lookback optimization
    enable_bayesian_optimization: bool = True
    spline_degree: int = 3
    penalty_weight: float = 0.1
    use_log_space: bool = True
    
    # Hierarchical shrinkage
    enable_hierarchical_shrinkage: bool = True
    n_samples: int = 2000  # MCMC samples
    warmup: int = 1000
    
    # Stability testing
    enable_stability_test: bool = True
    stability_threshold: float = 0.7  # Minimum stability score (<=30% sign flips)
    sign_flip_tolerance: float = 0.1  # Sign flip tolerance
    
    # Data availability requirements
    min_data_availability: float = 0.95  # Minimum data availability for book features
    book_dependent_families: List[str] = field(default_factory=lambda: [
        "order_flow", "microstructure", "liquidity"
    ])
    
    # HDI requirements
    max_hdi_width: float = 0.7  # Maximum HDI width in log-space
    min_utility_threshold: float = 0.0


@dataclass
class InteractionConfig:
    """Configuration for interaction feature generation."""
    # Interaction settings
    max_interactions: int = 15
    require_both_parents: bool = True  # Both parents must be selected
    interaction_types: List[str] = field(default_factory=lambda: [
        "multiplication", "division", "addition", "subtraction"
    ])
    
    # Parent selection
    min_parent_utility: float = 0.1  # Minimum utility for parent features
    max_correlation: float = 0.8  # Maximum correlation between parents
    
    # Interaction evaluation
    evaluate_interactions: bool = True
    interaction_utility_threshold: float = 0.05


@dataclass
class FinalSelectionConfig:
    """Configuration for final model-level selection."""
    # Stability selection
    enable_stability_selection: bool = True
    stability_threshold: float = 0.6  # Selection frequency threshold
    n_bootstrap_samples: int = 100
    
    # FDR control
    enable_fdr_control: bool = True
    fdr_q_value: float = 0.15  # FDR q-value (10-20%)
    
    # Group heredity
    enable_group_heredity: bool = True
    min_parents_required: int = 1  # At least one parent must be kept
    prefer_both_parents: bool = True
    
    # Final feature count
    target_feature_count: int = 45
    min_feature_count: int = 30
    max_feature_count: int = 60
    
    # Model settings
    model_type: str = "lightgbm"
    max_depth: int = 4
    learning_rate: float = 0.1
    n_estimators: int = 100


@dataclass
class DataDrivenFeatureSelectionConfig:
    """Main configuration for data-driven feature selection."""
    # Phase configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    final_selection: FinalSelectionConfig = field(default_factory=FinalSelectionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    
    # General settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_progress_tracking: bool = True
    save_intermediate_results: bool = True
    
    # Feature bank integration
    feature_bank_path: str = "src/feature_generation"
    exclude_categories: List[str] = field(default_factory=lambda: [
        "autoencoder", "cross_timeframe", "interaction"
    ])
    exclude_generators: List[str] = field(default_factory=lambda: [
        "bid_ask", "bidask", "regime_"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'phase1': self.phase1.__dict__,
            'phase2': self.phase2.__dict__,
            'interaction': self.interaction.__dict__,
            'final_selection': self.final_selection.__dict__,
            'budget': self.budget.__dict__,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'memory_limit_gb': self.memory_limit_gb,
            'enable_caching': self.enable_caching,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'log_level': self.log_level,
            'enable_progress_tracking': self.enable_progress_tracking,
            'save_intermediate_results': self.save_intermediate_results,
            'feature_bank_path': self.feature_bank_path,
            'exclude_categories': self.exclude_categories,
            'exclude_generators': self.exclude_generators
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataDrivenFeatureSelectionConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'phase1' in config_dict:
            config.phase1 = Phase1Config(**config_dict['phase1'])
        if 'phase2' in config_dict:
            config.phase2 = Phase2Config(**config_dict['phase2'])
        if 'interaction' in config_dict:
            config.interaction = InteractionConfig(**config_dict['interaction'])
        if 'final_selection' in config_dict:
            config.final_selection = FinalSelectionConfig(**config_dict['final_selection'])
        if 'budget' in config_dict:
            config.budget = BudgetConfig(**config_dict['budget'])
        
        # Update general settings
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ['phase1', 'phase2', 'interaction', 'final_selection', 'budget']:
                setattr(config, key, value)
        
        return config


def create_development_config() -> DataDrivenFeatureSelectionConfig:
    """Create development configuration (fast, less thorough)."""
    config = DataDrivenFeatureSelectionConfig()
    
    # Phase 1: Faster probes
    config.phase1.probe_days = 15
    config.phase1.subset_ratio = 0.2
    config.phase1.momentum_lookbacks = [5, 10]
    config.phase1.volatility_lookbacks = [6, 10]
    config.phase1.rsi_lookbacks = [7, 10]
    
    # Phase 2: Faster optimization
    config.phase2.n_samples = 500
    config.phase2.warmup = 250
    
    # Budget: More relaxed
    config.budget.max_features_pre_selection = 80
    config.budget.max_final_features = 40
    
    # Final selection: Faster
    config.final_selection.n_bootstrap_samples = 50
    config.final_selection.target_feature_count = 30
    
    return config


def create_production_config() -> DataDrivenFeatureSelectionConfig:
    """Create production configuration (thorough, robust)."""
    config = DataDrivenFeatureSelectionConfig()
    
    # Phase 1: More thorough probes
    config.phase1.probe_days = 25
    config.phase1.subset_ratio = 0.4
    config.phase1.momentum_lookbacks = [5, 8, 12, 16, 20]
    config.phase1.volatility_lookbacks = [6, 10, 14, 18, 22]
    config.phase1.rsi_lookbacks = [7, 10, 14, 18, 21]
    
    # Phase 2: More thorough optimization
    config.phase2.n_samples = 3000
    config.phase2.warmup = 1500
    
    # Budget: Strict constraints
    config.budget.max_features_pre_selection = 120
    config.budget.max_final_features = 60
    
    # Final selection: More thorough
    config.final_selection.n_bootstrap_samples = 200
    config.final_selection.target_feature_count = 45
    
    return config


def create_custom_config(
    phase1_overrides: Optional[Dict[str, Any]] = None,
    phase2_overrides: Optional[Dict[str, Any]] = None,
    budget_overrides: Optional[Dict[str, Any]] = None,
    final_selection_overrides: Optional[Dict[str, Any]] = None
) -> DataDrivenFeatureSelectionConfig:
    """Create custom configuration with overrides."""
    config = DataDrivenFeatureSelectionConfig()
    
    if phase1_overrides:
        for key, value in phase1_overrides.items():
            if hasattr(config.phase1, key):
                setattr(config.phase1, key, value)
    
    if phase2_overrides:
        for key, value in phase2_overrides.items():
            if hasattr(config.phase2, key):
                setattr(config.phase2, key, value)
    
    if budget_overrides:
        for key, value in budget_overrides.items():
            if hasattr(config.budget, key):
                setattr(config.budget, key, value)
    
    if final_selection_overrides:
        for key, value in final_selection_overrides.items():
            if hasattr(config.final_selection, key):
                setattr(config.final_selection, key, value)
    
    return config