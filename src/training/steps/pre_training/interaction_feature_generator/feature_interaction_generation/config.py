"""
Enhanced Configuration System for Data-Driven Lookback Optimization

This module provides comprehensive configuration for the three-stage Bayesian optimization
system with extensive enhancements including matrix operations, hardware optimization,
and comprehensive validation.

Key Features:
- Extensive configuration validation and error handling
- Matrix operations and hardware optimization settings
- ML utilities integration configuration
- Comprehensive logging and monitoring settings
- Performance optimization parameters
- Data quality and validation settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import yaml
import os
import logging
from pathlib import Path

# Import comprehensive utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_error, tprint_warning, tprint_success,
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)

# Import math validation
try:
    from src.utils.math_validation import (
        validate_finite, validate_positive, validate_range,
        safe_divide, safe_log, safe_sqrt
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    def validate_finite(x, name="value"): return x
    def validate_positive(x, name="value"): return x
    def validate_range(x, min_val=None, max_val=None, name="value"): return x
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def safe_log(x, default=0.0): return __import__('numpy').log(x) if x > 0 else default
    def safe_sqrt(x, default=0.0): return __import__('numpy').sqrt(x) if x >= 0 else default

# Set up logging
logger = logging.getLogger(__name__)


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


class LogLevel(Enum):
    """Logging levels for comprehensive monitoring."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class HardwareOptimizationMode(Enum):
    """Hardware optimization modes."""
    NONE = "none"
    CPU_ONLY = "cpu_only"
    MEMORY_ONLY = "memory_only"
    GPU_ONLY = "gpu_only"
    FULL = "full"


class MatrixOperationMode(Enum):
    """Matrix operation optimization modes."""
    NONE = "none"
    VECTORIZED = "vectorized"
    BATCH = "batch"
    GPU_ACCELERATED = "gpu_accelerated"
    FULL = "full"


@dataclass
class LoggingConfig:
    """Comprehensive logging configuration."""
    level: LogLevel = LogLevel.INFO
    enable_tprint: bool = True
    enable_file_logging: bool = True
    log_file_path: Optional[str] = None
    enable_performance_logging: bool = True
    enable_debug_logging: bool = False
    log_memory_usage: bool = True
    log_gpu_usage: bool = True
    log_matrix_operations: bool = True
    log_ml_operations: bool = True
    structured_logging: bool = False
    log_correlation_id: bool = True
    
    def __post_init__(self):
        """Validate logging configuration."""
        if self.log_file_path is None:
            self.log_file_path = "lookback_optimization.log"
        
        # Ensure log file directory exists
        if self.enable_file_logging:
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tprint_debug(f"📁 Log file directory created: {log_path.parent}")


@dataclass
class HardwareOptimizationConfig:
    """Hardware optimization configuration."""
    mode: HardwareOptimizationMode = HardwareOptimizationMode.FULL
    enable_m1_gpu: bool = True
    enable_m1_memory: bool = True
    enable_m1_cpu: bool = True
    memory_checkpoint_interval: float = 1.0  # seconds
    gpu_memory_fraction: float = 0.8
    cpu_threads: int = 4
    enable_memory_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    memory_optimization_threshold: float = 0.8  # 80% memory usage
    enable_automatic_cleanup: bool = True
    
    def __post_init__(self):
        """Validate hardware configuration."""
        if MATH_VALIDATION_AVAILABLE:
            self.gpu_memory_fraction = validate_range(
                self.gpu_memory_fraction, 0.0, 1.0, "gpu_memory_fraction"
            )
            self.memory_optimization_threshold = validate_range(
                self.memory_optimization_threshold, 0.0, 1.0, "memory_optimization_threshold"
            )
            self.cpu_threads = validate_positive(self.cpu_threads, "cpu_threads")


@dataclass
class MatrixOperationsConfig:
    """Matrix operations configuration."""
    mode: MatrixOperationMode = MatrixOperationMode.FULL
    enable_vectorized_processing: bool = True
    enable_batch_processing: bool = True
    enable_gpu_acceleration: bool = True
    batch_size: int = 1000
    vectorization_threshold: int = 100  # minimum size for vectorization
    enable_sparse_operations: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient_mode: bool = True
    enable_operation_caching: bool = True
    cache_size_mb: int = 100
    
    def __post_init__(self):
        """Validate matrix operations configuration."""
        if MATH_VALIDATION_AVAILABLE:
            self.batch_size = validate_positive(self.batch_size, "batch_size")
            self.vectorization_threshold = validate_positive(self.vectorization_threshold, "vectorization_threshold")
            self.max_workers = validate_positive(self.max_workers, "max_workers")
            self.cache_size_mb = validate_positive(self.cache_size_mb, "cache_size_mb")


@dataclass
class MLUtilitiesConfig:
    """ML utilities configuration."""
    enable_bayesian_optimization: bool = True
    enable_feature_selection: bool = True
    enable_data_leakage_detection: bool = True
    enable_lookahead_bias_detection: bool = True
    enable_hyperparameter_optimization: bool = True
    enable_model_validation: bool = True
    enable_out_of_fold_prediction: bool = True
    bayesian_n_trials: int = 100
    bayesian_timeout: int = 300  # seconds
    feature_selection_method: str = "mutual_info"
    data_leakage_threshold: float = 0.1
    lookahead_bias_threshold: float = 0.05
    cross_validation_folds: int = 5
    enable_purged_cv: bool = True
    embargo_percentage: float = 0.1
    
    def __post_init__(self):
        """Validate ML utilities configuration."""
        if MATH_VALIDATION_AVAILABLE:
            self.bayesian_n_trials = validate_positive(self.bayesian_n_trials, "bayesian_n_trials")
            self.bayesian_timeout = validate_positive(self.bayesian_timeout, "bayesian_timeout")
            self.data_leakage_threshold = validate_range(
                self.data_leakage_threshold, 0.0, 1.0, "data_leakage_threshold"
            )
            self.lookahead_bias_threshold = validate_range(
                self.lookahead_bias_threshold, 0.0, 1.0, "lookahead_bias_threshold"
            )
            self.cross_validation_folds = validate_positive(self.cross_validation_folds, "cross_validation_folds")
            self.embargo_percentage = validate_range(
                self.embargo_percentage, 0.0, 1.0, "embargo_percentage"
            )


@dataclass
class DataQualityConfig:
    """Data quality and validation configuration."""
    enable_quality_validation: bool = True
    min_data_points: int = 1000
    max_missing_percentage: float = 0.1
    max_outlier_percentage: float = 0.05
    enable_anomaly_detection: bool = True
    enable_trend_validation: bool = True
    enable_volatility_validation: bool = True
    quality_score_threshold: float = 0.7
    enable_automatic_cleaning: bool = True
    cleaning_method: str = "interpolate"  # interpolate, forward_fill, backward_fill, drop
    
    def __post_init__(self):
        """Validate data quality configuration."""
        if MATH_VALIDATION_AVAILABLE:
            self.min_data_points = validate_positive(self.min_data_points, "min_data_points")
            self.max_missing_percentage = validate_range(
                self.max_missing_percentage, 0.0, 1.0, "max_missing_percentage"
            )
            self.max_outlier_percentage = validate_range(
                self.max_outlier_percentage, 0.0, 1.0, "max_outlier_percentage"
            )
            self.quality_score_threshold = validate_range(
                self.quality_score_threshold, 0.0, 1.0, "quality_score_threshold"
            )


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
    """Enhanced main configuration for the lookback optimization system with comprehensive utilities."""
    
    # Core components
    penalties: CostPenalties = field(default_factory=CostPenalties)
    search_grids: SearchGrids = field(default_factory=SearchGrids)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    spline: SplineConfig = field(default_factory=SplineConfig)
    hac: HACConfig = field(default_factory=HACConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Enhanced configuration components
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareOptimizationConfig = field(default_factory=HardwareOptimizationConfig)
    matrix_operations: MatrixOperationsConfig = field(default_factory=MatrixOperationsConfig)
    ml_utilities: MLUtilitiesConfig = field(default_factory=MLUtilitiesConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    
    # Runtime settings
    enable_parallel: bool = True
    n_workers: int = 4
    memory_limit_gb: float = 8.0
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Enhanced logging and monitoring
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    output_dir: str = "lookback_optimization_results"
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    enable_matrix_operation_logging: bool = True
    enable_ml_operation_logging: bool = True
    
    # Performance optimization
    enable_optimization: bool = True
    optimization_level: str = "full"  # none, basic, full
    enable_caching: bool = True
    enable_compression: bool = True
    compression_level: int = 6
    
    # Data processing
    enable_data_validation: bool = True
    enable_quality_checks: bool = True
    enable_anomaly_detection: bool = True
    enable_automatic_cleaning: bool = True
    
    def __post_init__(self):
        """Validate and initialize configuration."""
        tprint_debug("🔧 Initializing Enhanced Lookback Optimization Configuration...")
        
        # Validate numeric values
        if MATH_VALIDATION_AVAILABLE:
            self.n_workers = validate_positive(self.n_workers, "n_workers")
            self.memory_limit_gb = validate_positive(self.memory_limit_gb, "memory_limit_gb")
            self.cache_size = validate_positive(self.cache_size, "cache_size")
            self.cache_ttl_seconds = validate_positive(self.cache_ttl_seconds, "cache_ttl_seconds")
            self.compression_level = validate_range(self.compression_level, 1, 9, "compression_level")
        
        # Ensure output directory exists
        if self.save_intermediate_results:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            tprint_debug(f"📁 Output directory created: {output_path}")
        
        # Log configuration summary
        tprint_info("📊 Configuration Summary:")
        tprint_info(f"   - Logging level: {self.log_level}")
        tprint_info(f"   - Parallel processing: {'✅' if self.enable_parallel else '❌'}")
        tprint_info(f"   - Workers: {self.n_workers}")
        tprint_info(f"   - Memory limit: {self.memory_limit_gb} GB")
        tprint_info(f"   - Hardware optimization: {self.hardware.mode.value}")
        tprint_info(f"   - Matrix operations: {self.matrix_operations.mode.value}")
        tprint_info(f"   - ML utilities: {'✅' if self.ml_utilities.enable_bayesian_optimization else '❌'}")
        tprint_info(f"   - Data quality validation: {'✅' if self.data_quality.enable_quality_validation else '❌'}")
        
        tprint_success("✅ Enhanced configuration initialized successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire enhanced configuration to dictionary."""
        return {
            # Core components
            'penalties': self.penalties.to_dict(),
            'search_grids': self.search_grids.to_dict(),
            'hysteresis': self.hysteresis.to_dict(),
            'spline': self.spline.to_dict(),
            'hac': self.hac.to_dict(),
            'cv': self.cv.to_dict(),
            'hierarchical': self.hierarchical.to_dict(),
            'export': self.export.to_dict(),
            
            # Enhanced configuration components
            'logging': {
                'level': self.logging.level.value,
                'enable_tprint': self.logging.enable_tprint,
                'enable_file_logging': self.logging.enable_file_logging,
                'log_file_path': self.logging.log_file_path,
                'enable_performance_logging': self.logging.enable_performance_logging,
                'enable_debug_logging': self.logging.enable_debug_logging,
                'log_memory_usage': self.logging.log_memory_usage,
                'log_gpu_usage': self.logging.log_gpu_usage,
                'log_matrix_operations': self.logging.log_matrix_operations,
                'log_ml_operations': self.logging.log_ml_operations,
                'structured_logging': self.logging.structured_logging,
                'log_correlation_id': self.logging.log_correlation_id
            },
            'hardware': {
                'mode': self.hardware.mode.value,
                'enable_m1_gpu': self.hardware.enable_m1_gpu,
                'enable_m1_memory': self.hardware.enable_m1_memory,
                'enable_m1_cpu': self.hardware.enable_m1_cpu,
                'memory_checkpoint_interval': self.hardware.memory_checkpoint_interval,
                'gpu_memory_fraction': self.hardware.gpu_memory_fraction,
                'cpu_threads': self.hardware.cpu_threads,
                'enable_memory_monitoring': self.hardware.enable_memory_monitoring,
                'enable_gpu_monitoring': self.hardware.enable_gpu_monitoring,
                'memory_optimization_threshold': self.hardware.memory_optimization_threshold,
                'enable_automatic_cleanup': self.hardware.enable_automatic_cleanup
            },
            'matrix_operations': {
                'mode': self.matrix_operations.mode.value,
                'enable_vectorized_processing': self.matrix_operations.enable_vectorized_processing,
                'enable_batch_processing': self.matrix_operations.enable_batch_processing,
                'enable_gpu_acceleration': self.matrix_operations.enable_gpu_acceleration,
                'batch_size': self.matrix_operations.batch_size,
                'vectorization_threshold': self.matrix_operations.vectorization_threshold,
                'enable_sparse_operations': self.matrix_operations.enable_sparse_operations,
                'enable_parallel_processing': self.matrix_operations.enable_parallel_processing,
                'max_workers': self.matrix_operations.max_workers,
                'memory_efficient_mode': self.matrix_operations.memory_efficient_mode,
                'enable_operation_caching': self.matrix_operations.enable_operation_caching,
                'cache_size_mb': self.matrix_operations.cache_size_mb
            },
            'ml_utilities': {
                'enable_bayesian_optimization': self.ml_utilities.enable_bayesian_optimization,
                'enable_feature_selection': self.ml_utilities.enable_feature_selection,
                'enable_data_leakage_detection': self.ml_utilities.enable_data_leakage_detection,
                'enable_lookahead_bias_detection': self.ml_utilities.enable_lookahead_bias_detection,
                'enable_hyperparameter_optimization': self.ml_utilities.enable_hyperparameter_optimization,
                'enable_model_validation': self.ml_utilities.enable_model_validation,
                'enable_out_of_fold_prediction': self.ml_utilities.enable_out_of_fold_prediction,
                'bayesian_n_trials': self.ml_utilities.bayesian_n_trials,
                'bayesian_timeout': self.ml_utilities.bayesian_timeout,
                'feature_selection_method': self.ml_utilities.feature_selection_method,
                'data_leakage_threshold': self.ml_utilities.data_leakage_threshold,
                'lookahead_bias_threshold': self.ml_utilities.lookahead_bias_threshold,
                'cross_validation_folds': self.ml_utilities.cross_validation_folds,
                'enable_purged_cv': self.ml_utilities.enable_purged_cv,
                'embargo_percentage': self.ml_utilities.embargo_percentage
            },
            'data_quality': {
                'enable_quality_validation': self.data_quality.enable_quality_validation,
                'min_data_points': self.data_quality.min_data_points,
                'max_missing_percentage': self.data_quality.max_missing_percentage,
                'max_outlier_percentage': self.data_quality.max_outlier_percentage,
                'enable_anomaly_detection': self.data_quality.enable_anomaly_detection,
                'enable_trend_validation': self.data_quality.enable_trend_validation,
                'enable_volatility_validation': self.data_quality.enable_volatility_validation,
                'quality_score_threshold': self.data_quality.quality_score_threshold,
                'enable_automatic_cleaning': self.data_quality.enable_automatic_cleaning,
                'cleaning_method': self.data_quality.cleaning_method
            },
            
            # Runtime settings
            'enable_parallel': self.enable_parallel,
            'n_workers': self.n_workers,
            'memory_limit_gb': self.memory_limit_gb,
            'cache_size': self.cache_size,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            
            # Enhanced logging and monitoring
            'log_level': self.log_level,
            'save_intermediate_results': self.save_intermediate_results,
            'output_dir': self.output_dir,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'enable_gpu_monitoring': self.enable_gpu_monitoring,
            'enable_matrix_operation_logging': self.enable_matrix_operation_logging,
            'enable_ml_operation_logging': self.enable_ml_operation_logging,
            
            # Performance optimization
            'enable_optimization': self.enable_optimization,
            'optimization_level': self.optimization_level,
            'enable_caching': self.enable_caching,
            'enable_compression': self.enable_compression,
            'compression_level': self.compression_level,
            
            # Data processing
            'enable_data_validation': self.enable_data_validation,
            'enable_quality_checks': self.enable_quality_checks,
            'enable_anomaly_detection': self.enable_anomaly_detection,
            'enable_automatic_cleaning': self.enable_automatic_cleaning
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


def create_optimized_config() -> LookbackOptimizationConfig:
    """Create optimized configuration for maximum performance."""
    tprint_info("🚀 Creating optimized configuration for maximum performance...")
    
    config = LookbackOptimizationConfig()
    
    # Optimize hardware settings
    config.hardware.mode = HardwareOptimizationMode.FULL
    config.hardware.enable_m1_gpu = True
    config.hardware.enable_m1_memory = True
    config.hardware.enable_m1_cpu = True
    config.hardware.cpu_threads = 8
    config.hardware.gpu_memory_fraction = 0.9
    
    # Optimize matrix operations
    config.matrix_operations.mode = MatrixOperationMode.FULL
    config.matrix_operations.batch_size = 2000
    config.matrix_operations.max_workers = 8
    config.matrix_operations.enable_operation_caching = True
    config.matrix_operations.cache_size_mb = 200
    
    # Optimize ML utilities
    config.ml_utilities.bayesian_n_trials = 200
    config.ml_utilities.bayesian_timeout = 600
    config.ml_utilities.cross_validation_folds = 10
    
    # Optimize data quality
    config.data_quality.quality_score_threshold = 0.8
    config.data_quality.enable_automatic_cleaning = True
    
    # Optimize runtime settings
    config.n_workers = 8
    config.memory_limit_gb = 16.0
    config.cache_size = 2000
    config.optimization_level = "full"
    
    tprint_success("✅ Optimized configuration created")
    return config


def create_minimal_config() -> LookbackOptimizationConfig:
    """Create minimal configuration for testing and development."""
    tprint_info("🧪 Creating minimal configuration for testing...")
    
    config = LookbackOptimizationConfig()
    
    # Minimal hardware settings
    config.hardware.mode = HardwareOptimizationMode.CPU_ONLY
    config.hardware.enable_m1_gpu = False
    config.hardware.cpu_threads = 2
    
    # Minimal matrix operations
    config.matrix_operations.mode = MatrixOperationMode.VECTORIZED
    config.matrix_operations.batch_size = 100
    config.matrix_operations.max_workers = 2
    config.matrix_operations.enable_operation_caching = False
    
    # Minimal ML utilities
    config.ml_utilities.bayesian_n_trials = 20
    config.ml_utilities.bayesian_timeout = 60
    config.ml_utilities.cross_validation_folds = 3
    
    # Minimal runtime settings
    config.n_workers = 2
    config.memory_limit_gb = 2.0
    config.cache_size = 100
    config.optimization_level = "basic"
    
    tprint_success("✅ Minimal configuration created")
    return config


def validate_config(config: LookbackOptimizationConfig) -> bool:
    """Validate configuration for consistency and correctness."""
    tprint_info("🔍 Validating configuration...")
    
    try:
        # Validate numeric ranges
        if MATH_VALIDATION_AVAILABLE:
            validate_positive(config.n_workers, "n_workers")
            validate_positive(config.memory_limit_gb, "memory_limit_gb")
            validate_positive(config.cache_size, "cache_size")
            validate_range(config.compression_level, 1, 9, "compression_level")
        
        # Validate configuration consistency
        if config.n_workers > 16:
            tprint_warning("⚠️ High number of workers may cause resource contention")
        
        if config.memory_limit_gb < 1.0:
            tprint_warning("⚠️ Low memory limit may cause performance issues")
        
        if config.hardware.gpu_memory_fraction > 0.95:
            tprint_warning("⚠️ Very high GPU memory fraction may cause OOM errors")
        
        # Validate output directory
        if config.save_intermediate_results:
            output_path = Path(config.output_dir)
            if not output_path.exists():
                tprint_warning(f"⚠️ Output directory does not exist: {output_path}")
        
        tprint_success("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Configuration validation failed: {e}")
        return False


def load_config_from_file(file_path: str) -> LookbackOptimizationConfig:
    """Load configuration from YAML file."""
    tprint_info(f"📁 Loading configuration from file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = LookbackOptimizationConfig.from_dict(config_dict)
        
        # Validate loaded configuration
        if validate_config(config):
            tprint_success("✅ Configuration loaded and validated successfully")
            return config
        else:
            tprint_error("❌ Configuration validation failed")
            raise ValueError("Invalid configuration file")
            
    except Exception as e:
        tprint_error(f"❌ Failed to load configuration from file: {e}")
        raise


def save_config_to_file(config: LookbackOptimizationConfig, file_path: str) -> None:
    """Save configuration to YAML file."""
    tprint_info(f"💾 Saving configuration to file: {file_path}")
    
    try:
        config_dict = config.to_dict()
        
        # Ensure directory exists
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        tprint_success("✅ Configuration saved successfully")
        
    except Exception as e:
        tprint_error(f"❌ Failed to save configuration to file: {e}")
        raise