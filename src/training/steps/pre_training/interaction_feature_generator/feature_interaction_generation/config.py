"""
Configuration system for data-driven lookback optimization with cost-aware selection.

This module provides comprehensive configuration for the three-stage Bayesian optimization
system that replaces hardcoded lookback ceilings with data-driven inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import yaml
import os


class OptimizationMode(Enum):
    """Optimization modes for lookback selection."""
    DISCRETE = "discrete"
    BLEND = "blend"
    DISCRETE_OR_BLEND = "discrete_or_blend"


class FamilyType(Enum):
    """Feature family types for lookback optimization."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VWAP_ROLL = "vwap_roll"
    RSI = "rsi"
    AUTOCORR = "autocorr"
    GK = "gk"


@dataclass
class CostPenalties:
    """Cost penalty configuration for lookback optimization."""
    lambda_cost: float = 0.05        # Penalty for CPU cost (latency impact)
    lambda_stale: float = 0.05       # Penalty for staleness (update lag)
    lambda_uncertainty: float = 0.10 # Penalty for estimation risk (HAC SE)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'lambda_cost': self.lambda_cost,
            'lambda_stale': self.lambda_stale,
            'lambda_uncertainty': self.lambda_uncertainty
        }


@dataclass
class SearchGrids:
    """Search grid configuration for different feature families."""
    momentum_bars: List[int] = field(default_factory=lambda: [5, 12, 24, 48, 96, 192])
    sigma_halflife: List[int] = field(default_factory=lambda: [6, 12, 18, 36, 72, 144])
    gk_window_bars: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 96])
    rsi_period: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    autocorr_window: List[int] = field(default_factory=lambda: [6, 12, 24, 48])
    vwap_roll_bars: List[int] = field(default_factory=lambda: [12, 36])
    
    def get_family_grid(self, family: FamilyType) -> List[int]:
        """Get search grid for specific family."""
        mapping = {
            FamilyType.MOMENTUM: self.momentum_bars,
            FamilyType.VOLATILITY: self.sigma_halflife,
            FamilyType.GK: self.gk_window_bars,
            FamilyType.RSI: self.rsi_period,
            FamilyType.AUTOCORR: self.autocorr_window,
            FamilyType.VWAP_ROLL: self.vwap_roll_bars
        }
        return mapping.get(family, [])
    
    def to_dict(self) -> Dict[str, List[int]]:
        """Convert to dictionary for serialization."""
        return {
            'momentum_bars': self.momentum_bars,
            'sigma_halflife': self.sigma_halflife,
            'gk_window_bars': self.gk_window_bars,
            'rsi_period': self.rsi_period,
            'autocorr_window': self.autocorr_window,
            'vwap_roll_bars': self.vwap_roll_bars
        }


@dataclass
class HysteresisConfig:
    """Hysteresis configuration for lookback stability."""
    min_delta_log_l: float = 0.2        # ≈ 22% change in log lookback
    min_delta_ic_sigma: float = 0.25    # Minimum IC improvement in sigma units
    max_hdi_width: float = 4.0          # Maximum HDI width for discrete choice
    min_fold_match_rate: float = 0.6    # Minimum fold match rate for stability
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'min_delta_log_l': self.min_delta_log_l,
            'min_delta_ic_sigma': self.min_delta_ic_sigma,
            'max_hdi_width': self.max_hdi_width,
            'min_fold_match_rate': self.min_fold_match_rate
        }


@dataclass
class SplineConfig:
    """Spline fitting configuration for IC surface estimation."""
    n_knots: int = 4
    degree: int = 3
    penalty_weight: float = 1.0
    use_log_space: bool = True
    min_data_points: int = 6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_knots': self.n_knots,
            'degree': self.degree,
            'penalty_weight': self.penalty_weight,
            'use_log_space': self.use_log_space,
            'min_data_points': self.min_data_points
        }


@dataclass
class HACConfig:
    """HAC (Heteroskedasticity and Autocorrelation Consistent) configuration."""
    lag_method: str = "sqrt_t"  # "sqrt_t", "fixed", "aic", "bic"
    fixed_lag: Optional[int] = None
    max_lag: int = 50
    kernel: str = "bartlett"  # "bartlett", "parzen", "quadratic"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'lag_method': self.lag_method,
            'fixed_lag': self.fixed_lag,
            'max_lag': self.max_lag,
            'kernel': self.kernel
        }


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration for purged walk-forward validation."""
    n_folds: int = 5
    purging_period: int = 5  # Bars to purge around each split
    embargo_period: int = 2  # Bars to embargo after each split
    min_train_size: int = 1000  # Minimum training samples
    min_test_size: int = 200   # Minimum test samples
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return {
            'n_folds': self.n_folds,
            'purging_period': self.purging_period,
            'embargo_period': self.embargo_period,
            'min_train_size': self.min_train_size,
            'min_test_size': self.min_test_size
        }


@dataclass
class HierarchicalConfig:
    """Hierarchical Bayesian shrinkage configuration."""
    use_variational: bool = True  # Use ADVI instead of NUTS
    n_samples: int = 1000
    n_tuning: int = 500
    target_accept: float = 0.8
    max_treedepth: int = 10
    adapt_delta: float = 0.8
    
    # Prior hyperparameters
    mu_prior_mean: float = 0.0
    mu_prior_std: float = 2.0
    tau_prior_scale: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'use_variational': self.use_variational,
            'n_samples': self.n_samples,
            'n_tuning': self.n_tuning,
            'target_accept': self.target_accept,
            'max_treedepth': self.max_treedepth,
            'adapt_delta': self.adapt_delta,
            'mu_prior_mean': self.mu_prior_mean,
            'mu_prior_std': self.mu_prior_std,
            'tau_prior_scale': self.tau_prior_scale
        }


@dataclass
class ExportConfig:
    """Export configuration for production deployment."""
    mode: OptimizationMode = OptimizationMode.DISCRETE_OR_BLEND
    max_windows_per_family: int = 3
    max_total_features: int = 120  # Pre-selection cap
    max_interactions: int = 15     # Interaction cap
    max_p99_latency_ms: int = 50   # Latency constraint
    
    # Allowed discrete windows for production
    allowed_windows: Dict[FamilyType, List[int]] = field(default_factory=lambda: {
        FamilyType.MOMENTUM: [5, 12, 24],
        FamilyType.VOLATILITY: [6, 12, 18],
        FamilyType.GK: [6, 12, 24],
        FamilyType.VWAP_ROLL: [6, 12],
        FamilyType.RSI: [7, 14],
        FamilyType.AUTOCORR: [6, 12]
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mode': self.mode.value,
            'max_windows_per_family': self.max_windows_per_family,
            'max_total_features': self.max_total_features,
            'max_interactions': self.max_interactions,
            'max_p99_latency_ms': self.max_p99_latency_ms,
            'allowed_windows': {k.value: v for k, v in self.allowed_windows.items()}
        }


@dataclass
class LookbackOptimizationConfig:
    """Main configuration for the lookback optimization system."""
    
    # Core components
    penalties: CostPenalties = field(default_factory=CostPenalties)
    search_grids: SearchGrids = field(default_factory=SearchGrids)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    spline: SplineConfig = field(default_factory=SplineConfig)
    hac: HACConfig = field(default_factory=HACConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Runtime settings
    enable_parallel: bool = True
    n_workers: int = 4
    memory_limit_gb: float = 8.0
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    output_dir: str = "lookback_optimization_results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            'penalties': self.penalties.to_dict(),
            'search_grids': self.search_grids.to_dict(),
            'hysteresis': self.hysteresis.to_dict(),
            'spline': self.spline.to_dict(),
            'hac': self.hac.to_dict(),
            'cv': self.cv.to_dict(),
            'hierarchical': self.hierarchical.to_dict(),
            'export': self.export.to_dict(),
            'enable_parallel': self.enable_parallel,
            'n_workers': self.n_workers,
            'memory_limit_gb': self.memory_limit_gb,
            'cache_size': self.cache_size,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'log_level': self.log_level,
            'save_intermediate_results': self.save_intermediate_results,
            'output_dir': self.output_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LookbackOptimizationConfig':
        """Create configuration from dictionary."""
        # Extract nested configurations
        penalties = CostPenalties(**config_dict.get('penalties', {}))
        search_grids = SearchGrids(**config_dict.get('search_grids', {}))
        hysteresis = HysteresisConfig(**config_dict.get('hysteresis', {}))
        spline = SplineConfig(**config_dict.get('spline', {}))
        hac = HACConfig(**config_dict.get('hac', {}))
        cv = CrossValidationConfig(**config_dict.get('cv', {}))
        hierarchical = HierarchicalConfig(**config_dict.get('hierarchical', {}))
        
        # Handle export config with enum conversion
        export_dict = config_dict.get('export', {})
        if 'mode' in export_dict and isinstance(export_dict['mode'], str):
            export_dict['mode'] = OptimizationMode(export_dict['mode'])
        
        # Convert allowed_windows back to enum keys
        if 'allowed_windows' in export_dict:
            allowed_windows = {}
            for k, v in export_dict['allowed_windows'].items():
                allowed_windows[FamilyType(k)] = v
            export_dict['allowed_windows'] = allowed_windows
        
        export = ExportConfig(**export_dict)
        
        # Create main config
        return cls(
            penalties=penalties,
            search_grids=search_grids,
            hysteresis=hysteresis,
            spline=spline,
            hac=hac,
            cv=cv,
            hierarchical=hierarchical,
            export=export,
            enable_parallel=config_dict.get('enable_parallel', True),
            n_workers=config_dict.get('n_workers', 4),
            memory_limit_gb=config_dict.get('memory_limit_gb', 8.0),
            cache_size=config_dict.get('cache_size', 1000),
            cache_ttl_seconds=config_dict.get('cache_ttl_seconds', 3600),
            log_level=config_dict.get('log_level', 'INFO'),
            save_intermediate_results=config_dict.get('save_intermediate_results', True),
            output_dir=config_dict.get('output_dir', 'lookback_optimization_results')
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LookbackOptimizationConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate penalty weights
        if self.penalties.lambda_cost < 0:
            issues.append("lambda_cost must be non-negative")
        if self.penalties.lambda_stale < 0:
            issues.append("lambda_stale must be non-negative")
        if self.penalties.lambda_uncertainty < 0:
            issues.append("lambda_uncertainty must be non-negative")
        
        # Validate search grids
        for family in FamilyType:
            grid = self.search_grids.get_family_grid(family)
            if not grid or len(grid) < 3:
                issues.append(f"Search grid for {family.value} must have at least 3 points")
            if any(x <= 0 for x in grid):
                issues.append(f"All values in {family.value} grid must be positive")
        
        # Validate CV config
        if self.cv.n_folds < 2:
            issues.append("n_folds must be at least 2")
        if self.cv.purging_period < 0:
            issues.append("purging_period must be non-negative")
        if self.cv.embargo_period < 0:
            issues.append("embargo_period must be non-negative")
        
        # Validate export config
        if self.export.max_total_features <= 0:
            issues.append("max_total_features must be positive")
        if self.export.max_interactions <= 0:
            issues.append("max_interactions must be positive")
        if self.export.max_p99_latency_ms <= 0:
            issues.append("max_p99_latency_ms must be positive")
        
        return issues


def create_default_config() -> LookbackOptimizationConfig:
    """Create default configuration with production-ready settings."""
    return LookbackOptimizationConfig()


def create_development_config() -> LookbackOptimizationConfig:
    """Create development configuration with relaxed constraints."""
    config = LookbackOptimizationConfig()
    
    # Relaxed search grids for development
    config.search_grids.momentum_bars = [5, 12, 24]
    config.search_grids.sigma_halflife = [6, 12, 18]
    config.search_grids.gk_window_bars = [6, 12, 24]
    config.search_grids.rsi_period = [7, 14]
    config.search_grids.autocorr_window = [6, 12]
    config.search_grids.vwap_roll_bars = [12]
    
    # Fewer CV folds for faster development
    config.cv.n_folds = 3
    
    # Reduced hierarchical samples
    config.hierarchical.n_samples = 500
    config.hierarchical.n_tuning = 250
    
    return config


def create_production_config() -> LookbackOptimizationConfig:
    """Create production configuration with strict constraints."""
    config = LookbackOptimizationConfig()
    
    # Strict cost penalties
    config.penalties.lambda_cost = 0.1
    config.penalties.lambda_stale = 0.1
    config.penalties.lambda_uncertainty = 0.15
    
    # Conservative hysteresis
    config.hysteresis.min_delta_log_l = 0.3
    config.hysteresis.min_delta_ic_sigma = 0.5
    config.hysteresis.max_hdi_width = 3.0
    config.hysteresis.min_fold_match_rate = 0.7
    
    # More CV folds for stability
    config.cv.n_folds = 7
    
    # More hierarchical samples
    config.hierarchical.n_samples = 2000
    config.hierarchical.n_tuning = 1000
    
    return config