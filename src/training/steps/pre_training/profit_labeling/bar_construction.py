"""
Event-Based Bar Construction for Volatility-Aware Labeling

This module implements event-based bar construction utilities that create
volatility-normalized bars for more robust profit labeling.

Key Features:
- Event-based bar construction using volume, volatility, and time triggers
- Volatility-normalized bar sizes for consistent signal quality
- Adaptive bar construction based on market conditions
- Integration with existing ML optimization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import copy
from contextlib import contextmanager

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview, tprint_data_format
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range
)
from src.utils.math_validation import MathValidation

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig
    from src.utils.hardware.enhanced_cpu_optimizer import EnhancedCPUOptimizer
    from src.utils.hardware.advanced_memory_optimizer import AdvancedMemoryOptimizer
    from src.utils.hardware.enhanced_caching_system import EnhancedCachingSystem
    from src.utils.ml_common.optimization.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available, using grid search")


class BarTriggerType(Enum):
    """Enumeration of bar construction trigger types."""
    VOLUME = "volume"  # Volume-based triggers
    VOLATILITY = "volatility"  # Volatility-based triggers
    TIME = "time"  # Time-based triggers
    HYBRID = "hybrid"  # Combined approach


class OptimizationPhase(Enum):
    """Enumeration of optimization phases."""
    INITIALIZATION = "initialization"
    PARAMETER_SEARCH = "parameter_search"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"


@dataclass
class TemporaryOptimizationConfig:
    """Temporary configuration object used during optimization phases."""
    
    # Phase identification
    phase: OptimizationPhase = OptimizationPhase.INITIALIZATION
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
    
    # Parameter bounds for optimization
    volume_threshold_bounds: Tuple[float, float] = (100.0, 10000.0)
    volatility_threshold_bounds: Tuple[float, float] = (0.001, 0.05)
    volume_multiplier_bounds: Tuple[float, float] = (1.0, 3.0)
    volatility_multiplier_bounds: Tuple[float, float] = (1.0, 4.0)
    
    # Optimization constraints
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    early_stopping_patience: int = 10
    
    # Validation settings
    validation_split: float = 0.2
    cross_validation_folds: int = 3
    min_validation_samples: int = 100
    
    # Performance tracking
    enable_performance_tracking: bool = True
    track_memory_usage: bool = True
    track_computation_time: bool = True
    
    # Quality thresholds
    min_quality_score: float = 0.3
    max_quality_score: float = 1.0
    target_quality_score: float = 0.7
    
    # Adaptive parameters
    enable_adaptive_bounds: bool = True
    adaptive_window_size: int = 50
    bounds_adjustment_factor: float = 0.1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.volume_threshold_bounds[0] >= self.volume_threshold_bounds[1]:
            raise ValueError("volume_threshold_bounds must have lower < upper")
        if self.volatility_threshold_bounds[0] >= self.volatility_threshold_bounds[1]:
            raise ValueError("volatility_threshold_bounds must have lower < upper")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.min_quality_score >= self.max_quality_score:
            raise ValueError("min_quality_score must be less than max_quality_score")


@dataclass
class TemporaryParameterConfig:
    """Temporary parameter configuration during optimization iterations."""
    
    # Current parameter values
    volume_threshold: float = 1000.0
    volatility_threshold: float = 0.01
    volume_multiplier: float = 1.5
    volatility_multiplier: float = 2.0
    
    # Parameter metadata
    iteration_number: int = 0
    parameter_id: str = ""
    parent_config_id: str = ""
    
    # Performance metrics
    quality_score: float = 0.0
    computation_time: float = 0.0
    memory_usage: float = 0.0
    
    # Validation results
    validation_score: float = 0.0
    cross_validation_scores: List[float] = field(default_factory=list)
    validation_std: float = 0.0
    
    # Convergence tracking
    improvement_delta: float = 0.0
    convergence_rate: float = 0.0
    is_converged: bool = False
    
    # Quality assessment
    bar_count: int = 0
    avg_bar_duration: float = 0.0
    avg_bar_volume: float = 0.0
    avg_bar_volatility: float = 0.0
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite score from multiple metrics."""
        if weights is None:
            weights = {
                'quality_score': 0.4,
                'validation_score': 0.3,
                'computation_efficiency': 0.2,
                'stability': 0.1
            }
        
        # Computation efficiency (inverse of time and memory)
        efficiency = 1.0 / (1.0 + self.computation_time + self.memory_usage)
        
        # Stability (inverse of validation standard deviation)
        stability = 1.0 / (1.0 + self.validation_std)
        
        composite = (
            weights['quality_score'] * self.quality_score +
            weights['validation_score'] * self.validation_score +
            weights['computation_efficiency'] * efficiency +
            weights['stability'] * stability
        )
        
        return composite


@dataclass
class TemporaryValidationConfig:
    """Temporary configuration for validation during optimization."""
    
    # Validation parameters
    validation_method: str = "time_series_split"
    test_size: float = 0.2
    gap_size: int = 0  # Gap between train and test to prevent leakage
    
    # Cross-validation settings
    cv_strategy: str = "time_series_cv"
    n_splits: int = 3
    test_size_cv: float = 0.2
    
    # Quality metrics to evaluate
    primary_metric: str = "sharpe_ratio"
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "information_ratio", "max_drawdown", "hit_rate", "profit_factor"
    ])
    
    # Statistical significance testing
    enable_statistical_tests: bool = True
    significance_level: float = 0.05
    min_samples_for_test: int = 30
    
    # Robustness testing
    enable_robustness_tests: bool = True
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    perturbation_types: List[str] = field(default_factory=lambda: [
        "gaussian_noise", "outlier_injection", "missing_data"
    ])
    
    # Performance constraints
    max_computation_time: float = 300.0  # 5 minutes
    max_memory_usage: float = 1024.0  # 1GB
    min_validation_samples: int = 50
    
    def __post_init__(self):
        """Validate validation configuration."""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")


@dataclass
class TemporaryMemoryConfig:
    """Temporary configuration for memory management during optimization."""
    
    # Memory limits
    max_memory_usage: float = 2048.0  # MB
    memory_warning_threshold: float = 1536.0  # MB
    memory_critical_threshold: float = 1920.0  # MB
    
    # Garbage collection settings
    enable_aggressive_gc: bool = True
    gc_frequency: int = 10  # Every N iterations
    gc_threshold: float = 0.8  # Trigger GC when memory usage exceeds this
    
    # Data management
    enable_data_compression: bool = True
    compression_level: int = 6
    enable_lazy_loading: bool = True
    
    # Cache management
    enable_result_caching: bool = True
    max_cache_size: int = 100
    cache_eviction_policy: str = "lru"
    
    # Monitoring
    enable_memory_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    log_memory_usage: bool = True
    
    def __post_init__(self):
        """Validate memory configuration."""
        if self.max_memory_usage <= 0:
            raise ValueError("max_memory_usage must be positive")
        if not 0 < self.memory_warning_threshold < self.memory_critical_threshold < self.max_memory_usage:
            raise ValueError("Memory thresholds must be in ascending order")


@dataclass
class TemporaryPerformanceConfig:
    """Temporary configuration for performance monitoring during optimization."""
    
    # Performance tracking
    enable_timing: bool = True
    enable_profiling: bool = False
    enable_memory_profiling: bool = True
    
    # Timing granularity
    track_function_calls: bool = True
    track_iteration_times: bool = True
    track_phase_times: bool = True
    
    # Profiling settings
    profile_frequency: int = 1  # Profile every N iterations
    profile_depth: int = 3  # Call stack depth
    profile_memory: bool = True
    
    # Performance thresholds
    max_iteration_time: float = 60.0  # seconds
    max_phase_time: float = 300.0  # seconds
    warning_time_threshold: float = 30.0  # seconds
    
    # Optimization settings
    enable_early_stopping: bool = True
    patience_iterations: int = 5
    improvement_threshold: float = 0.001
    
    # Reporting
    enable_progress_reporting: bool = True
    report_frequency: int = 10  # Every N iterations
    enable_detailed_logging: bool = False
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_iteration_time <= 0:
            raise ValueError("max_iteration_time must be positive")
        if self.patience_iterations < 1:
            raise ValueError("patience_iterations must be at least 1")


@dataclass
class BarConstructionConfig:
    """Configuration for event-based bar construction."""
    
    # Trigger settings
    trigger_type: BarTriggerType = BarTriggerType.HYBRID
    
    # Volume-based triggers
    volume_threshold: float = 1000.0  # Minimum volume for bar completion
    volume_multiplier: float = 1.5  # Volume multiplier for adaptive sizing
    
    # Volatility-based triggers
    volatility_threshold: float = 0.01  # Minimum volatility for bar completion
    volatility_multiplier: float = 2.0  # Volatility multiplier for adaptive sizing
    
    # Time-based triggers
    max_bar_duration: timedelta = timedelta(minutes=5)  # Maximum bar duration
    min_bar_duration: timedelta = timedelta(seconds=30)  # Minimum bar duration
    
    # Adaptive sizing
    enable_adaptive_sizing: bool = True
    adaptive_window: int = 20  # Window for adaptive parameter calculation
    
    # Quality checks
    min_bar_samples: int = 10
    max_price_change_ratio: float = 0.1  # Maximum price change ratio within a bar
    
    # Data-driven optimization
    enable_optimization: bool = True
    optimization_metric: str = "sharpe_ratio"  # Optimization target metric
    
    def _validate_config(self) -> None:
        """Validate bar construction configuration parameters."""
        if self.volume_threshold <= 0:
            raise ValueError("volume_threshold must be positive")
        if self.volatility_threshold <= 0:
            raise ValueError("volatility_threshold must be positive")
        if self.max_bar_duration <= self.min_bar_duration:
            raise ValueError("max_bar_duration must be greater than min_bar_duration")
        if self.min_bar_samples < 1:
            raise ValueError("min_bar_samples must be at least 1")
        if self.max_price_change_ratio <= 0:
            raise ValueError("max_price_change_ratio must be positive")


@dataclass
class BarConstructionResult:
    """Result container for bar construction."""
    
    # Core results
    constructed_bars: pd.DataFrame
    construction_metadata: Dict[str, Any]
    
    # Statistics
    total_bars: int = 0
    avg_bar_duration: float = 0.0
    avg_bar_volume: float = 0.0
    avg_bar_volatility: float = 0.0
    
    # Quality metrics
    bar_quality_score: float = 0.0
    volatility_consistency: float = 0.0
    volume_consistency: float = 0.0
    
    # Metadata
    config_used: BarConstructionConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class EventBasedBarConstructor:
    """
    Event-Based Bar Constructor for Volatility-Aware Labeling
    
    This class implements sophisticated bar construction that creates
    volatility-normalized bars for more robust profit labeling.
    
    Key Features:
    1. **Event-Based Construction**: Bars are created based on volume, volatility, and time triggers
    2. **Volatility Normalization**: Bar sizes are normalized by volatility for consistency
    3. **Adaptive Sizing**: Bar parameters adapt to changing market conditions
    4. **Quality Validation**: Comprehensive bar quality assessment
    5. **Data-Driven Optimization**: Parameters optimized using historical data
    """
    
    def __init__(self, config: Optional[BarConstructionConfig] = None):
        """Initialize event-based bar constructor."""
        self.config = config or BarConstructionConfig()
        self.logger = logging.getLogger('EventBasedBarConstructor')
        
        # Validate configuration
        self.config._validate_config()
        
        # Initialize optimization if available
        if BAYESIAN_OPTIMIZER_AVAILABLE and self.config.enable_optimization:
            self.optimizer = BayesianTPEOptimizer()
            self.vectorization_manager = UnifiedVectorizationManager(
                VectorizationConfig(
                    enable_vectorization=True,
                    vectorization_method="numpy",
                    batch_size=1000,
                    enable_parallel_processing=True,
                    enable_optimization=True,
                    enable_gpu_acceleration=True,
                    memory_efficient=True
                )
            )
            self.cpu_optimizer = EnhancedCPUOptimizer()
            self.memory_optimizer = AdvancedMemoryOptimizer()
            self.caching_system = EnhancedCachingSystem()
            self.vectorbt_optimizer = VectorBTRollingOptimizer()
            tprint_info("   → Bayesian optimization: Available")
            tprint_info("   → VectorBTRollingOptimizer: Available")
            tprint_info("   → Hardware acceleration: Available")
        else:
            self.optimizer = None
            self.vectorization_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.caching_system = None
            self.vectorbt_optimizer = None
            tprint_warning("   → Bayesian optimization: Not available, using fixed parameters")
        
        tprint_info("📊 Event-Based Bar Constructor initialized")
        tprint_info(f"   → Trigger type: {self.config.trigger_type.value}")
        tprint_info(f"   → Volume threshold: {self.config.volume_threshold}")
        tprint_info(f"   → Volatility threshold: {self.config.volatility_threshold}")
        tprint_info(f"   → Adaptive sizing: {self.config.enable_adaptive_sizing}")
    
    def construct_bars(self, tick_data: pd.DataFrame) -> BarConstructionResult:
        """
        Construct event-based bars from tick data.
        
        Args:
            tick_data: Tick data with OHLCV and timestamp columns
            
        Returns:
            BarConstructionResult with constructed bars and metadata
        """
        start_time = datetime.now()
        tprint_info("📊 Constructing event-based bars")
        
        # Initialize result container
        result = BarConstructionResult(
            constructed_bars=pd.DataFrame(),
            construction_metadata={},
            config_used=self.config
        )
        
        try:
            # Validate input data
            if not self._validate_input_data(tick_data):
                return result
            
            # Optimize parameters if enabled
            if self.config.enable_optimization and self.optimizer:
                tprint_info("🔧 Step 1: Optimizing bar construction parameters")
                optimized_config = self._optimize_parameters(tick_data)
                self.config = optimized_config
            
            # Calculate adaptive parameters
            if self.config.enable_adaptive_sizing:
                tprint_info("📈 Step 2: Calculating adaptive parameters")
                adaptive_params = self._calculate_adaptive_parameters(tick_data)
            else:
                adaptive_params = self._get_default_parameters()
            
            # Construct bars based on trigger type
            tprint_info("🔨 Step 3: Constructing bars")
            if self.config.trigger_type == BarTriggerType.VOLUME:
                bars = self._construct_volume_based_bars(tick_data, adaptive_params)
            elif self.config.trigger_type == BarTriggerType.VOLATILITY:
                bars = self._construct_volatility_based_bars(tick_data, adaptive_params)
            elif self.config.trigger_type == BarTriggerType.TIME:
                bars = self._construct_time_based_bars(tick_data, adaptive_params)
            else:  # HYBRID
                bars = self._construct_hybrid_bars(tick_data, adaptive_params)
            
            result.constructed_bars = bars
            
            # Calculate statistics and quality metrics
            tprint_info("📊 Step 4: Calculating statistics and quality metrics")
            stats = self._calculate_bar_statistics(bars)
            result.total_bars = stats['total_bars']
            result.avg_bar_duration = stats['avg_bar_duration']
            result.avg_bar_volume = stats['avg_bar_volume']
            result.avg_bar_volatility = stats['avg_bar_volatility']
            
            quality_metrics = self._calculate_bar_quality(bars)
            result.bar_quality_score = quality_metrics['quality_score']
            result.volatility_consistency = quality_metrics['volatility_consistency']
            result.volume_consistency = quality_metrics['volume_consistency']
            
            # Store construction metadata
            result.construction_metadata = {
                'trigger_type': self.config.trigger_type.value,
                'adaptive_params': adaptive_params,
                'optimization_enabled': self.config.enable_optimization,
                'bars_constructed': len(bars)
            }
            
        except Exception as e:
            tprint_error(f"❌ Bar construction failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Bar construction completed")
        tprint_info(f"   → Bars constructed: {result.total_bars}")
        tprint_info(f"   → Avg duration: {result.avg_bar_duration:.2f}s")
        tprint_info(f"   → Quality score: {result.bar_quality_score:.3f}")
        
        return result
    
    def _validate_input_data(self, tick_data: pd.DataFrame) -> bool:
        """Validate input tick data."""
        try:
            # Check if DataFrame is empty
            if tick_data.empty:
                tprint_warning("⚠️ Input tick data is empty")
                return False
            
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(tick_data.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check minimum samples
            if len(tick_data) < self.config.min_bar_samples:
                tprint_warning(f"⚠️ Insufficient samples: {len(tick_data)} < {self.config.min_bar_samples}")
                return False
            
            # Check for non-finite values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            if tick_data[numeric_columns].isnull().any().any():
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if not np.isfinite(tick_data[numeric_columns].values).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _optimize_parameters(self, tick_data: pd.DataFrame) -> BarConstructionConfig:
        """Optimize bar construction parameters using historical data with temporary configurations."""
        try:
            if not self.optimizer:
                return self.config
            
            # Create temporary optimization configuration
            optimization_config = create_temporary_optimization_config(
                strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
                max_iterations=50,
                validation_split=0.2
            )
            
            # Create temporary validation configuration
            validation_config = create_temporary_validation_config(
                validation_method="time_series_split",
                n_splits=3,
                enable_robustness=True
            )
            
            # Create temporary memory configuration
            memory_config = create_temporary_memory_config(
                max_memory_usage=2048.0,
                enable_aggressive_gc=True,
                enable_result_caching=True
            )
            
            # Create temporary performance configuration
            performance_config = create_temporary_performance_config(
                enable_profiling=False,
                enable_early_stopping=True,
                patience_iterations=5
            )
            
            # Validate all temporary configurations
            if not validate_temporary_configs(
                optimization_config, 
                TemporaryParameterConfig(), 
                validation_config, 
                memory_config, 
                performance_config
            ):
                tprint_warning("⚠️ Configuration validation failed, using default parameters")
                return self.config
            
            # Create parameter space from optimization config
            param_space = create_optimization_parameter_space(optimization_config)
            
            # Track optimization performance
            optimization_start_time = datetime.now()
            iteration_count = 0
            best_score = 0.0
            best_params = None
            
            # Define objective function with temporary configurations
            def objective(params):
                nonlocal iteration_count, best_score, best_params
                iteration_count += 1
                
                # Create temporary parameter configuration
                param_config = create_temporary_parameter_config(
                    volume_threshold=params['volume_threshold'],
                    volatility_threshold=params['volatility_threshold'],
                    volume_multiplier=params['volume_multiplier'],
                    volatility_multiplier=params['volatility_multiplier'],
                    iteration_number=iteration_count
                )
                
                # Use temporary optimization config context
                with temporary_optimization_config(
                    self.config, 
                    optimization_config, 
                    param_config
                ) as temp_config:
                    # Create temporary constructor
                    temp_constructor = EventBasedBarConstructor(temp_config)
                    
                    # Construct bars and evaluate quality
                    result = temp_constructor.construct_bars(tick_data)
                    
                    # Calculate composite score
                    param_config.quality_score = result.bar_quality_score
                    param_config.bar_count = result.total_bars
                    param_config.avg_bar_duration = result.avg_bar_duration
                    param_config.avg_bar_volume = result.avg_bar_volume
                    param_config.avg_bar_volatility = result.avg_bar_volatility
                    
                    # Calculate composite score
                    composite_score = param_config.calculate_composite_score()
                    
                    # Track best parameters
                    if composite_score > best_score:
                        best_score = composite_score
                        best_params = params.copy()
                    
                    # Log progress
                    if iteration_count % 10 == 0:
                        tprint_info(f"   → Iteration {iteration_count}: Score = {composite_score:.4f}")
                    
                    return composite_score
            
            # Run optimization with temporary configurations
            with temporary_memory_config(memory_config) as mem_config, \
                 temporary_performance_config(performance_config) as perf_config:
                
                best_params = self.optimizer.optimize(
                    objective_function=objective,
                    param_space=param_space,
                    n_trials=optimization_config.max_iterations,
                    random_state=42
                )
            
            # Calculate optimization time
            optimization_time = (datetime.now() - optimization_start_time).total_seconds()
            
            # Update config with optimized parameters
            optimized_config = BarConstructionConfig(
                trigger_type=self.config.trigger_type,
                volume_threshold=best_params['volume_threshold'],
                volatility_threshold=best_params['volatility_threshold'],
                volume_multiplier=best_params['volume_multiplier'],
                volatility_multiplier=best_params['volatility_multiplier'],
                max_bar_duration=self.config.max_bar_duration,
                min_bar_duration=self.config.min_bar_duration,
                enable_adaptive_sizing=self.config.enable_adaptive_sizing,
                adaptive_window=self.config.adaptive_window,
                min_bar_samples=self.config.min_bar_samples,
                max_price_change_ratio=self.config.max_price_change_ratio,
                enable_optimization=False
            )
            
            tprint_success("✅ Parameter optimization completed")
            tprint_info(f"   → Iterations: {iteration_count}")
            tprint_info(f"   → Best score: {best_score:.4f}")
            tprint_info(f"   → Optimization time: {optimization_time:.2f}s")
            tprint_info(f"   → Volume threshold: {best_params['volume_threshold']:.2f}")
            tprint_info(f"   → Volatility threshold: {best_params['volatility_threshold']:.4f}")
            
            return optimized_config
            
        except Exception as e:
            tprint_warning(f"⚠️ Parameter optimization failed: {e}")
            return self.config
    
    def _calculate_adaptive_parameters(self, tick_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate adaptive parameters based on historical data."""
        try:
            # Calculate rolling statistics
            window = min(self.config.adaptive_window, len(tick_data) // 2)
            
            # Volume statistics
            volume_mean = tick_data['volume'].rolling(window=window).mean().iloc[-1]
            volume_std = tick_data['volume'].rolling(window=window).std().iloc[-1]
            
            # Volatility statistics
            returns = tick_data['close'].pct_change().dropna()
            volatility_mean = returns.rolling(window=window).std().iloc[-1]
            volatility_std = returns.rolling(window=window).std().std()
            
            # Adaptive thresholds
            adaptive_volume_threshold = max(
                volume_mean + 0.5 * volume_std,
                self.config.volume_threshold
            )
            
            adaptive_volatility_threshold = max(
                volatility_mean + 0.5 * volatility_std,
                self.config.volatility_threshold
            )
            
            # Adaptive multipliers based on market conditions
            volume_multiplier = 1.0 + (volume_std / volume_mean) if volume_mean > 0 else 1.0
            volatility_multiplier = 1.0 + (volatility_std / volatility_mean) if volatility_mean > 0 else 1.0
            
            return {
                'volume_threshold': adaptive_volume_threshold,
                'volatility_threshold': adaptive_volatility_threshold,
                'volume_multiplier': min(volume_multiplier, 3.0),  # Cap at 3.0
                'volatility_multiplier': min(volatility_multiplier, 4.0)  # Cap at 4.0
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating adaptive parameters: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameters when adaptive calculation fails."""
        return {
            'volume_threshold': self.config.volume_threshold,
            'volatility_threshold': self.config.volatility_threshold,
            'volume_multiplier': self.config.volume_multiplier,
            'volatility_multiplier': self.config.volatility_multiplier
        }
    
    def _construct_volume_based_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars based on volume triggers with vectorized operations."""
        try:
            # Check cache first
            cache_key = f"volume_bars_{len(tick_data)}_{params['volume_threshold']}_{params['volume_multiplier']}"
            if self.caching_system:
                cached_result = self.caching_system.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Use vectorized operations for better performance
            if self.vectorization_manager:
                tick_data = self.vectorization_manager.vectorize_data(tick_data)
            
            # Vectorized volume-based bar construction
            if self.vectorbt_optimizer:
                bars = self.vectorbt_optimizer.construct_volume_bars(
                    tick_data,
                    volume_threshold=params['volume_threshold'] * params['volume_multiplier'],
                    use_gpu=True
                )
            else:
                # Fallback to iterative method with optimizations
                bars = self._construct_volume_bars_iterative(tick_data, params)
            
            # Cache the result
            if self.caching_system:
                self.caching_system.set(cache_key, bars)
            
            return bars
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing volume-based bars: {e}")
            return pd.DataFrame()
    
    def _construct_volume_bars_iterative(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Fallback iterative method for volume-based bar construction."""
        bars = []
        current_bar = None
        cumulative_volume = 0.0
        
        for idx, row in tick_data.iterrows():
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                cumulative_volume = row['volume']
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
                current_bar['volume'] += row['volume']
                cumulative_volume += row['volume']
                
                # Check if volume threshold is reached
                if cumulative_volume >= params['volume_threshold'] * params['volume_multiplier']:
                    bars.append(current_bar)
                    current_bar = None
                    cumulative_volume = 0.0
        
        # Add final bar if exists
        if current_bar is not None:
            bars.append(current_bar)
        
        return pd.DataFrame(bars)
    
    def _construct_volatility_based_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars based on volatility triggers with vectorized operations."""
        try:
            # Check cache first
            cache_key = f"volatility_bars_{len(tick_data)}_{params['volatility_threshold']}_{params['volatility_multiplier']}"
            if self.caching_system:
                cached_result = self.caching_system.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Use vectorized operations for better performance
            if self.vectorization_manager:
                tick_data = self.vectorization_manager.vectorize_data(tick_data)
            
            # Vectorized volatility-based bar construction
            if self.vectorbt_optimizer:
                bars = self.vectorbt_optimizer.construct_volatility_bars(
                    tick_data,
                    volatility_threshold=params['volatility_threshold'] * params['volatility_multiplier'],
                    use_gpu=True
                )
            else:
                # Fallback to iterative method with optimizations
                bars = self._construct_volatility_bars_iterative(tick_data, params)
            
            # Cache the result
            if self.caching_system:
                self.caching_system.set(cache_key, bars)
            
            return bars
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing volatility-based bars: {e}")
            return pd.DataFrame()
    
    def _construct_volatility_bars_iterative(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Fallback iterative method for volatility-based bar construction."""
        bars = []
        current_bar = None
        bar_returns = []
        
        for idx, row in tick_data.iterrows():
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                bar_returns = []
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
                current_bar['volume'] += row['volume']
                
                # Calculate return
                return_val = (row['close'] - current_bar['open']) / current_bar['open']
                bar_returns.append(return_val)
                
                # Check if volatility threshold is reached
                if len(bar_returns) > 1:
                    volatility = np.std(bar_returns)
                    if volatility >= params['volatility_threshold'] * params['volatility_multiplier']:
                        bars.append(current_bar)
                        current_bar = None
                        bar_returns = []
        
        # Add final bar if exists
        if current_bar is not None:
            bars.append(current_bar)
        
        return pd.DataFrame(bars)
    
    def _construct_time_based_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars based on time triggers."""
        try:
            bars = []
            current_bar = None
            bar_start_time = None
            
            for idx, row in tick_data.iterrows():
                if current_bar is None:
                    # Start new bar
                    current_bar = {
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    bar_start_time = row['timestamp']
                else:
                    # Update current bar
                    current_bar['high'] = max(current_bar['high'], row['high'])
                    current_bar['low'] = min(current_bar['low'], row['low'])
                    current_bar['close'] = row['close']
                    current_bar['volume'] += row['volume']
                    
                    # Check if time threshold is reached
                    bar_duration = row['timestamp'] - bar_start_time
                    if bar_duration >= self.config.max_bar_duration:
                        bars.append(current_bar)
                        current_bar = None
                        bar_start_time = None
            
            # Add final bar if exists
            if current_bar is not None:
                bars.append(current_bar)
            
            return pd.DataFrame(bars)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing time-based bars: {e}")
            return pd.DataFrame()
    
    def _construct_hybrid_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars using hybrid approach with vectorized operations."""
        try:
            # Check cache first
            cache_key = f"hybrid_bars_{len(tick_data)}_{params['volume_threshold']}_{params['volatility_threshold']}"
            if self.caching_system:
                cached_result = self.caching_system.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Use vectorized operations for better performance
            if self.vectorization_manager:
                tick_data = self.vectorization_manager.vectorize_data(tick_data)
            
            # Vectorized hybrid bar construction
            if self.vectorbt_optimizer:
                bars = self.vectorbt_optimizer.construct_hybrid_bars(
                    tick_data,
                    volume_threshold=params['volume_threshold'] * params['volume_multiplier'],
                    volatility_threshold=params['volatility_threshold'] * params['volatility_multiplier'],
                    max_duration=self.config.max_bar_duration,
                    use_gpu=True
                )
            else:
                # Fallback to iterative method with optimizations
                bars = self._construct_hybrid_bars_iterative(tick_data, params)
            
            # Cache the result
            if self.caching_system:
                self.caching_system.set(cache_key, bars)
            
            return bars
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing hybrid bars: {e}")
            return pd.DataFrame()
    
    def _construct_hybrid_bars_iterative(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Fallback iterative method for hybrid bar construction."""
        bars = []
        current_bar = None
        cumulative_volume = 0.0
        bar_returns = []
        bar_start_time = None
        
        for idx, row in tick_data.iterrows():
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                cumulative_volume = row['volume']
                bar_returns = []
                bar_start_time = row['timestamp']
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
                current_bar['volume'] += row['volume']
                
                # Update tracking variables
                cumulative_volume += row['volume']
                return_val = (row['close'] - current_bar['open']) / current_bar['open']
                bar_returns.append(return_val)
                
                # Check multiple triggers
                volume_trigger = cumulative_volume >= params['volume_threshold'] * params['volume_multiplier']
                volatility_trigger = len(bar_returns) > 1 and np.std(bar_returns) >= params['volatility_threshold'] * params['volatility_multiplier']
                time_trigger = (row['timestamp'] - bar_start_time) >= self.config.max_bar_duration
                
                # Complete bar if any trigger is met
                if volume_trigger or volatility_trigger or time_trigger:
                    bars.append(current_bar)
                    current_bar = None
                    cumulative_volume = 0.0
                    bar_returns = []
                    bar_start_time = None
        
        # Add final bar if exists
        if current_bar is not None:
            bars.append(current_bar)
        
        return pd.DataFrame(bars)
    
    def _calculate_bar_statistics(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate bar construction statistics."""
        try:
            if bars.empty:
                return {
                    'total_bars': 0,
                    'avg_bar_duration': 0.0,
                    'avg_bar_volume': 0.0,
                    'avg_bar_volatility': 0.0
                }
            
            # Basic statistics
            total_bars = len(bars)
            
            # Duration statistics
            if 'timestamp' in bars.columns and len(bars) > 1:
                durations = bars['timestamp'].diff().dt.total_seconds().dropna()
                avg_duration = durations.mean() if not durations.empty else 0.0
            else:
                avg_duration = 0.0
            
            # Volume statistics
            avg_volume = bars['volume'].mean() if 'volume' in bars.columns else 0.0
            
            # Volatility statistics
            if 'open' in bars.columns and 'close' in bars.columns:
                returns = (bars['close'] - bars['open']) / bars['open']
                avg_volatility = returns.std() if not returns.empty else 0.0
            else:
                avg_volatility = 0.0
            
            return {
                'total_bars': total_bars,
                'avg_bar_duration': avg_duration,
                'avg_bar_volume': avg_volume,
                'avg_bar_volatility': avg_volatility
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating bar statistics: {e}")
            return {
                'total_bars': 0,
                'avg_bar_duration': 0.0,
                'avg_bar_volume': 0.0,
                'avg_bar_volatility': 0.0
            }
    
    def _calculate_bar_quality(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Calculate bar quality metrics."""
        try:
            if bars.empty:
                return {
                    'quality_score': 0.0,
                    'volatility_consistency': 0.0,
                    'volume_consistency': 0.0
                }
            
            # Quality score based on multiple factors
            quality_factors = []
            
            # Volume consistency
            if 'volume' in bars.columns and len(bars) > 1:
                volume_cv = bars['volume'].std() / bars['volume'].mean() if bars['volume'].mean() > 0 else 1.0
                volume_consistency = max(0.0, 1.0 - volume_cv)
                quality_factors.append(volume_consistency)
            else:
                volume_consistency = 0.0
                quality_factors.append(0.0)
            
            # Volatility consistency
            if 'open' in bars.columns and 'close' in bars.columns and len(bars) > 1:
                returns = (bars['close'] - bars['open']) / bars['open']
                volatility_cv = returns.std() / returns.mean() if returns.mean() > 0 else 1.0
                volatility_consistency = max(0.0, 1.0 - volatility_cv)
                quality_factors.append(volatility_consistency)
            else:
                volatility_consistency = 0.0
                quality_factors.append(0.0)
            
            # Price consistency (high-low range)
            if 'high' in bars.columns and 'low' in bars.columns and 'open' in bars.columns:
                price_ranges = (bars['high'] - bars['low']) / bars['open']
                range_cv = price_ranges.std() / price_ranges.mean() if price_ranges.mean() > 0 else 1.0
                price_consistency = max(0.0, 1.0 - range_cv)
                quality_factors.append(price_consistency)
            else:
                quality_factors.append(0.0)
            
            # Overall quality score
            quality_score = np.mean(quality_factors) if quality_factors else 0.0
            
            return {
                'quality_score': quality_score,
                'volatility_consistency': volatility_consistency,
                'volume_consistency': volume_consistency
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating bar quality: {e}")
            return {
                'quality_score': 0.0,
                'volatility_consistency': 0.0,
                'volume_consistency': 0.0
            }


# Context managers for temporary configurations
@contextmanager
def temporary_optimization_config(
    base_config: BarConstructionConfig,
    optimization_config: TemporaryOptimizationConfig,
    parameter_config: Optional[TemporaryParameterConfig] = None
):
    """Context manager for temporary optimization configuration."""
    original_config = copy.deepcopy(base_config)
    
    try:
        # Apply temporary parameter configuration if provided
        if parameter_config:
            base_config.volume_threshold = parameter_config.volume_threshold
            base_config.volatility_threshold = parameter_config.volatility_threshold
            base_config.volume_multiplier = parameter_config.volume_multiplier
            base_config.volatility_multiplier = parameter_config.volatility_multiplier
        
        # Disable optimization to prevent recursion
        base_config.enable_optimization = False
        
        yield base_config
        
    finally:
        # Restore original configuration
        base_config.__dict__.update(original_config.__dict__)


@contextmanager
def temporary_validation_config(
    validation_config: TemporaryValidationConfig,
    enable_robustness: bool = True
):
    """Context manager for temporary validation configuration."""
    original_robustness = validation_config.enable_robustness_tests
    
    try:
        # Apply robustness setting
        validation_config.enable_robustness_tests = enable_robustness
        
        yield validation_config
        
    finally:
        # Restore original setting
        validation_config.enable_robustness_tests = original_robustness


@contextmanager
def temporary_memory_config(
    memory_config: TemporaryMemoryConfig,
    enable_monitoring: bool = True
):
    """Context manager for temporary memory configuration."""
    original_monitoring = memory_config.enable_memory_monitoring
    
    try:
        # Apply monitoring setting
        memory_config.enable_memory_monitoring = enable_monitoring
        
        yield memory_config
        
    finally:
        # Restore original setting
        memory_config.enable_memory_monitoring = original_monitoring


@contextmanager
def temporary_performance_config(
    performance_config: TemporaryPerformanceConfig,
    enable_profiling: bool = False
):
    """Context manager for temporary performance configuration."""
    original_profiling = performance_config.enable_profiling
    
    try:
        # Apply profiling setting
        performance_config.enable_profiling = enable_profiling
        
        yield performance_config
        
    finally:
        # Restore original setting
        performance_config.enable_profiling = original_profiling


# Utility functions for temporary configurations
def create_temporary_optimization_config(
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION,
    max_iterations: int = 100,
    validation_split: float = 0.2
) -> TemporaryOptimizationConfig:
    """Create a temporary optimization configuration with specified parameters."""
    return TemporaryOptimizationConfig(
        strategy=strategy,
        max_iterations=max_iterations,
        validation_split=validation_split
    )


def create_temporary_parameter_config(
    volume_threshold: float = 1000.0,
    volatility_threshold: float = 0.01,
    volume_multiplier: float = 1.5,
    volatility_multiplier: float = 2.0,
    iteration_number: int = 0
) -> TemporaryParameterConfig:
    """Create a temporary parameter configuration with specified values."""
    return TemporaryParameterConfig(
        volume_threshold=volume_threshold,
        volatility_threshold=volatility_threshold,
        volume_multiplier=volume_multiplier,
        volatility_multiplier=volatility_multiplier,
        iteration_number=iteration_number,
        parameter_id=f"param_{iteration_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def create_temporary_validation_config(
    validation_method: str = "time_series_split",
    n_splits: int = 3,
    enable_robustness: bool = True
) -> TemporaryValidationConfig:
    """Create a temporary validation configuration with specified parameters."""
    return TemporaryValidationConfig(
        validation_method=validation_method,
        n_splits=n_splits,
        enable_robustness_tests=enable_robustness
    )


def create_temporary_memory_config(
    max_memory_usage: float = 2048.0,
    enable_aggressive_gc: bool = True,
    enable_result_caching: bool = True
) -> TemporaryMemoryConfig:
    """Create a temporary memory configuration with specified parameters."""
    return TemporaryMemoryConfig(
        max_memory_usage=max_memory_usage,
        enable_aggressive_gc=enable_aggressive_gc,
        enable_result_caching=enable_result_caching
    )


def create_temporary_performance_config(
    enable_profiling: bool = False,
    enable_early_stopping: bool = True,
    patience_iterations: int = 5
) -> TemporaryPerformanceConfig:
    """Create a temporary performance configuration with specified parameters."""
    return TemporaryPerformanceConfig(
        enable_profiling=enable_profiling,
        enable_early_stopping=enable_early_stopping,
        patience_iterations=patience_iterations
    )


def validate_temporary_configs(
    optimization_config: TemporaryOptimizationConfig,
    parameter_config: TemporaryParameterConfig,
    validation_config: TemporaryValidationConfig,
    memory_config: TemporaryMemoryConfig,
    performance_config: TemporaryPerformanceConfig
) -> bool:
    """Validate all temporary configurations for consistency."""
    try:
        # Validate individual configurations
        optimization_config.__post_init__()
        validation_config.__post_init__()
        memory_config.__post_init__()
        performance_config.__post_init__()
        
        # Cross-validation checks
        if optimization_config.validation_split != validation_config.test_size:
            tprint_warning("⚠️ Mismatch between optimization and validation split ratios")
        
        if optimization_config.max_iterations > performance_config.patience_iterations * 10:
            tprint_warning("⚠️ Max iterations much larger than patience iterations")
        
        if memory_config.max_memory_usage < 512.0:
            tprint_warning("⚠️ Memory limit may be too low for optimization")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Configuration validation failed: {e}")
        return False


def merge_temporary_configs(
    base_config: BarConstructionConfig,
    optimization_config: TemporaryOptimizationConfig,
    parameter_config: TemporaryParameterConfig
) -> BarConstructionConfig:
    """Merge temporary configurations into base configuration."""
    merged_config = copy.deepcopy(base_config)
    
    # Apply parameter values
    merged_config.volume_threshold = parameter_config.volume_threshold
    merged_config.volatility_threshold = parameter_config.volatility_threshold
    merged_config.volume_multiplier = parameter_config.volume_multiplier
    merged_config.volatility_multiplier = parameter_config.volatility_multiplier
    
    # Apply optimization settings
    merged_config.enable_optimization = False  # Prevent recursion
    
    return merged_config


def create_optimization_parameter_space(
    optimization_config: TemporaryOptimizationConfig
) -> Dict[str, Tuple[float, float]]:
    """Create parameter space for optimization based on configuration."""
    return {
        'volume_threshold': optimization_config.volume_threshold_bounds,
        'volatility_threshold': optimization_config.volatility_threshold_bounds,
        'volume_multiplier': optimization_config.volume_multiplier_bounds,
        'volatility_multiplier': optimization_config.volatility_multiplier_bounds
    }


def create_parameter_constraints(
    optimization_config: TemporaryOptimizationConfig
) -> List[Dict[str, Any]]:
    """Create parameter constraints for optimization."""
    constraints = []
    
    # Volume threshold constraints
    constraints.append({
        'name': 'volume_threshold_range',
        'type': 'range',
        'bounds': optimization_config.volume_threshold_bounds
    })
    
    # Volatility threshold constraints
    constraints.append({
        'name': 'volatility_threshold_range',
        'type': 'range',
        'bounds': optimization_config.volatility_threshold_bounds
    })
    
    # Multiplier constraints
    constraints.append({
        'name': 'volume_multiplier_range',
        'type': 'range',
        'bounds': optimization_config.volume_multiplier_bounds
    })
    
    constraints.append({
        'name': 'volatility_multiplier_range',
        'type': 'range',
        'bounds': optimization_config.volatility_multiplier_bounds
    })
    
    return constraints


# Convenience functions
def create_bar_construction_manager(config: Optional[BarConstructionConfig] = None) -> EventBasedBarConstructor:
    """Create bar construction manager with specified configuration."""
    return EventBasedBarConstructor(config)


def construct_bars(tick_data: pd.DataFrame,
                  config: Optional[BarConstructionConfig] = None) -> BarConstructionResult:
    """Construct bars with default configuration."""
    constructor = EventBasedBarConstructor(config)
    return constructor.construct_bars(tick_data)


class TemporaryConfigurationManager:
    """
    Manager class for handling temporary configurations during optimization.
    
    This class provides a centralized way to manage all temporary configurations
    used during the bar construction optimization process.
    """
    
    def __init__(self):
        """Initialize the temporary configuration manager."""
        self.optimization_config: Optional[TemporaryOptimizationConfig] = None
        self.parameter_config: Optional[TemporaryParameterConfig] = None
        self.validation_config: Optional[TemporaryValidationConfig] = None
        self.memory_config: Optional[TemporaryMemoryConfig] = None
        self.performance_config: Optional[TemporaryPerformanceConfig] = None
        
        self.configuration_history: List[Dict[str, Any]] = []
        self.optimization_metrics: Dict[str, List[float]] = {
            'quality_scores': [],
            'validation_scores': [],
            'computation_times': [],
            'memory_usage': []
        }
    
    def initialize_optimization(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        max_iterations: int = 100,
        validation_split: float = 0.2
    ) -> None:
        """Initialize all temporary configurations for optimization."""
        self.optimization_config = create_temporary_optimization_config(
            strategy=strategy,
            max_iterations=max_iterations,
            validation_split=validation_split
        )
        
        self.validation_config = create_temporary_validation_config(
            validation_method="time_series_split",
            n_splits=3,
            enable_robustness=True
        )
        
        self.memory_config = create_temporary_memory_config(
            max_memory_usage=2048.0,
            enable_aggressive_gc=True,
            enable_result_caching=True
        )
        
        self.performance_config = create_temporary_performance_config(
            enable_profiling=False,
            enable_early_stopping=True,
            patience_iterations=5
        )
        
        tprint_info("🔧 Temporary configuration manager initialized")
    
    def create_parameter_config(
        self,
        volume_threshold: float,
        volatility_threshold: float,
        volume_multiplier: float,
        volatility_multiplier: float,
        iteration_number: int = 0
    ) -> TemporaryParameterConfig:
        """Create a new parameter configuration."""
        self.parameter_config = create_temporary_parameter_config(
            volume_threshold=volume_threshold,
            volatility_threshold=volatility_threshold,
            volume_multiplier=volume_multiplier,
            volatility_multiplier=volatility_multiplier,
            iteration_number=iteration_number
        )
        
        return self.parameter_config
    
    def update_metrics(
        self,
        quality_score: float,
        validation_score: float,
        computation_time: float,
        memory_usage: float
    ) -> None:
        """Update optimization metrics."""
        self.optimization_metrics['quality_scores'].append(quality_score)
        self.optimization_metrics['validation_scores'].append(validation_score)
        self.optimization_metrics['computation_times'].append(computation_time)
        self.optimization_metrics['memory_usage'].append(memory_usage)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization metrics."""
        if not self.optimization_metrics['quality_scores']:
            return {}
        
        return {
            'total_iterations': len(self.optimization_metrics['quality_scores']),
            'best_quality_score': max(self.optimization_metrics['quality_scores']),
            'avg_quality_score': np.mean(self.optimization_metrics['quality_scores']),
            'best_validation_score': max(self.optimization_metrics['validation_scores']),
            'avg_validation_score': np.mean(self.optimization_metrics['validation_scores']),
            'total_computation_time': sum(self.optimization_metrics['computation_times']),
            'avg_computation_time': np.mean(self.optimization_metrics['computation_times']),
            'max_memory_usage': max(self.optimization_metrics['memory_usage']),
            'avg_memory_usage': np.mean(self.optimization_metrics['memory_usage'])
        }
    
    def check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_metrics['quality_scores']) < 10:
            return False
        
        recent_scores = self.optimization_metrics['quality_scores'][-10:]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < (self.optimization_config.convergence_tolerance if self.optimization_config else 1e-6)
    
    def should_early_stop(self) -> bool:
        """Check if optimization should stop early."""
        if not self.performance_config or not self.performance_config.enable_early_stopping:
            return False
        
        if len(self.optimization_metrics['quality_scores']) < self.performance_config.patience_iterations:
            return False
        
        recent_scores = self.optimization_metrics['quality_scores'][-self.performance_config.patience_iterations:]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.performance_config.improvement_threshold
    
    def cleanup(self) -> None:
        """Clean up temporary configurations and reset state."""
        self.optimization_config = None
        self.parameter_config = None
        self.validation_config = None
        self.memory_config = None
        self.performance_config = None
        
        self.configuration_history.clear()
        for key in self.optimization_metrics:
            self.optimization_metrics[key].clear()
        
        tprint_info("🧹 Temporary configuration manager cleaned up")


def create_temporary_configuration_manager() -> TemporaryConfigurationManager:
    """Create a new temporary configuration manager."""
    return TemporaryConfigurationManager()


def optimize_bar_construction_with_temporary_configs(
    tick_data: pd.DataFrame,
    base_config: Optional[BarConstructionConfig] = None,
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
) -> Tuple[BarConstructionResult, Dict[str, Any]]:
    """
    Optimize bar construction using temporary configurations.
    
    Args:
        tick_data: Input tick data
        base_config: Base configuration to optimize
        optimization_strategy: Strategy to use for optimization
        
    Returns:
        Tuple of (optimized_result, optimization_summary)
    """
    # Create configuration manager
    config_manager = create_temporary_configuration_manager()
    
    try:
        # Initialize optimization
        config_manager.initialize_optimization(strategy=optimization_strategy)
        
        # Create base constructor
        constructor = EventBasedBarConstructor(base_config)
        
        # Run optimization with temporary configurations
        result = constructor.construct_bars(tick_data)
        
        # Get optimization summary
        summary = config_manager.get_optimization_summary()
        
        return result, summary
        
    finally:
        # Cleanup
        config_manager.cleanup()


def create_adaptive_temporary_configs(
    market_data: pd.DataFrame,
    base_config: BarConstructionConfig
) -> Tuple[TemporaryOptimizationConfig, TemporaryValidationConfig, TemporaryMemoryConfig]:
    """
    Create adaptive temporary configurations based on market data characteristics.
    
    Args:
        market_data: Market data to analyze
        base_config: Base bar construction configuration
        
    Returns:
        Tuple of (optimization_config, validation_config, memory_config)
    """
    # Analyze market data characteristics
    data_length = len(market_data)
    volatility = market_data['close'].pct_change().std() if 'close' in market_data.columns else 0.01
    volume_mean = market_data['volume'].mean() if 'volume' in market_data.columns else 1000.0
    
    # Adaptive optimization configuration
    if data_length < 1000:
        max_iterations = 25
        validation_split = 0.3
    elif data_length < 5000:
        max_iterations = 50
        validation_split = 0.2
    else:
        max_iterations = 100
        validation_split = 0.15
    
    # Adjust parameter bounds based on volatility
    volatility_factor = min(max(volatility / 0.01, 0.5), 2.0)  # Scale between 0.5 and 2.0
    
    optimization_config = TemporaryOptimizationConfig(
        strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        max_iterations=max_iterations,
        validation_split=validation_split,
        volume_threshold_bounds=(volume_mean * 0.1, volume_mean * 10.0),
        volatility_threshold_bounds=(volatility * 0.1, volatility * 5.0),
        volume_multiplier_bounds=(1.0, 2.0 * volatility_factor),
        volatility_multiplier_bounds=(1.0, 3.0 * volatility_factor)
    )
    
    # Adaptive validation configuration
    validation_config = TemporaryValidationConfig(
        validation_method="time_series_split",
        n_splits=min(5, max(3, data_length // 1000)),
        test_size=validation_split,
        enable_robustness_tests=data_length > 2000,
        min_validation_samples=max(50, data_length // 20)
    )
    
    # Adaptive memory configuration
    estimated_memory = data_length * 0.001  # Rough estimate in MB
    memory_config = TemporaryMemoryConfig(
        max_memory_usage=max(1024.0, estimated_memory * 2),
        enable_aggressive_gc=data_length > 5000,
        enable_result_caching=data_length > 1000,
        max_cache_size=min(50, max(10, data_length // 100))
    )
    
    return optimization_config, validation_config, memory_config


def create_robust_temporary_configs(
    base_config: BarConstructionConfig,
    robustness_level: str = "medium"
) -> Tuple[TemporaryOptimizationConfig, TemporaryValidationConfig, TemporaryMemoryConfig]:
    """
    Create robust temporary configurations for challenging optimization scenarios.
    
    Args:
        base_config: Base bar construction configuration
        robustness_level: Level of robustness ("low", "medium", "high")
        
    Returns:
        Tuple of (optimization_config, validation_config, memory_config)
    """
    if robustness_level == "low":
        max_iterations = 25
        n_splits = 3
        enable_robustness = False
        max_memory = 1024.0
    elif robustness_level == "medium":
        max_iterations = 50
        n_splits = 5
        enable_robustness = True
        max_memory = 2048.0
    else:  # high
        max_iterations = 100
        n_splits = 7
        enable_robustness = True
        max_memory = 4096.0
    
    optimization_config = TemporaryOptimizationConfig(
        strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        max_iterations=max_iterations,
        convergence_tolerance=1e-8 if robustness_level == "high" else 1e-6,
        early_stopping_patience=15 if robustness_level == "high" else 10,
        enable_adaptive_bounds=True,
        adaptive_window_size=100 if robustness_level == "high" else 50
    )
    
    validation_config = TemporaryValidationConfig(
        validation_method="time_series_split",
        n_splits=n_splits,
        enable_robustness_tests=enable_robustness,
        enable_statistical_tests=robustness_level != "low",
        significance_level=0.01 if robustness_level == "high" else 0.05,
        noise_levels=[0.01, 0.05, 0.1] if robustness_level == "high" else [0.05],
        perturbation_types=["gaussian_noise", "outlier_injection"] if robustness_level == "high" else ["gaussian_noise"]
    )
    
    memory_config = TemporaryMemoryConfig(
        max_memory_usage=max_memory,
        enable_aggressive_gc=robustness_level != "low",
        enable_data_compression=robustness_level == "high",
        enable_lazy_loading=robustness_level == "high",
        enable_result_caching=True,
        max_cache_size=200 if robustness_level == "high" else 100
    )
    
    return optimization_config, validation_config, memory_config


def create_fast_temporary_configs(
    base_config: BarConstructionConfig
) -> Tuple[TemporaryOptimizationConfig, TemporaryValidationConfig, TemporaryMemoryConfig]:
    """
    Create fast temporary configurations for quick optimization scenarios.
    
    Args:
        base_config: Base bar construction configuration
        
    Returns:
        Tuple of (optimization_config, validation_config, memory_config)
    """
    optimization_config = TemporaryOptimizationConfig(
        strategy=OptimizationStrategy.GRID_SEARCH,
        max_iterations=20,
        convergence_tolerance=1e-4,
        early_stopping_patience=5,
        enable_adaptive_bounds=False,
        adaptive_window_size=20
    )
    
    validation_config = TemporaryValidationConfig(
        validation_method="time_series_split",
        n_splits=2,
        enable_robustness_tests=False,
        enable_statistical_tests=False,
        min_validation_samples=25
    )
    
    memory_config = TemporaryMemoryConfig(
        max_memory_usage=512.0,
        enable_aggressive_gc=False,
        enable_data_compression=False,
        enable_lazy_loading=False,
        enable_result_caching=False,
        max_cache_size=10
    )
    
    return optimization_config, validation_config, memory_config


def validate_temporary_configuration_compatibility(
    optimization_config: TemporaryOptimizationConfig,
    validation_config: TemporaryValidationConfig,
    memory_config: TemporaryMemoryConfig,
    performance_config: TemporaryPerformanceConfig
) -> List[str]:
    """
    Validate compatibility between temporary configurations.
    
    Args:
        optimization_config: Optimization configuration
        validation_config: Validation configuration
        memory_config: Memory configuration
        performance_config: Performance configuration
        
    Returns:
        List of compatibility warnings/errors
    """
    warnings = []
    
    # Check validation split consistency
    if abs(optimization_config.validation_split - validation_config.test_size) > 0.01:
        warnings.append("Validation split mismatch between optimization and validation configs")
    
    # Check iteration limits
    if optimization_config.max_iterations > 1000:
        warnings.append("Very high iteration count may cause performance issues")
    
    # Check memory constraints
    if memory_config.max_memory_usage < 256.0:
        warnings.append("Low memory limit may cause optimization failures")
    
    # Check performance constraints
    if performance_config.max_iteration_time < 10.0:
        warnings.append("Very low iteration time limit may cause premature termination")
    
    # Check validation sample requirements
    if validation_config.min_validation_samples > 1000:
        warnings.append("High minimum validation samples may limit data usage")
    
        return warnings


def create_optimization_context(
    base_config: BarConstructionConfig,
    market_data: pd.DataFrame,
    optimization_mode: str = "adaptive"
) -> Tuple[TemporaryConfigurationManager, BarConstructionConfig]:
    """
    Create a complete optimization context with appropriate temporary configurations.
    
    Args:
        base_config: Base bar construction configuration
        market_data: Market data for analysis
        optimization_mode: Mode for optimization ("adaptive", "robust", "fast")
        
    Returns:
        Tuple of (configuration_manager, optimized_base_config)
    """
    config_manager = create_temporary_configuration_manager()
    
    try:
        if optimization_mode == "adaptive":
            opt_config, val_config, mem_config = create_adaptive_temporary_configs(
                market_data, base_config
            )
        elif optimization_mode == "robust":
            opt_config, val_config, mem_config = create_robust_temporary_configs(
                base_config, "high"
            )
        elif optimization_mode == "fast":
            opt_config, val_config, mem_config = create_fast_temporary_configs(
                base_config
            )
        else:
            raise ValueError(f"Unknown optimization mode: {optimization_mode}")
        
        # Set configurations in manager
        config_manager.optimization_config = opt_config
        config_manager.validation_config = val_config
        config_manager.memory_config = mem_config
        
        # Create performance config
        config_manager.performance_config = create_temporary_performance_config(
            enable_profiling=(optimization_mode == "robust"),
            enable_early_stopping=True,
            patience_iterations=opt_config.early_stopping_patience
        )
        
        # Validate compatibility
        warnings = validate_temporary_configuration_compatibility(
            opt_config, val_config, mem_config, config_manager.performance_config
        )
        
        if warnings:
            for warning in warnings:
                tprint_warning(f"⚠️ {warning}")
        
        # Create optimized base config
        optimized_base_config = copy.deepcopy(base_config)
        optimized_base_config.enable_optimization = True
        
        return config_manager, optimized_base_config
        
    except Exception as e:
        tprint_error(f"❌ Failed to create optimization context: {e}")
        config_manager.cleanup()
        raise


def run_optimized_bar_construction(
    tick_data: pd.DataFrame,
    base_config: Optional[BarConstructionConfig] = None,
    optimization_mode: str = "adaptive",
    return_optimization_details: bool = False
) -> Union[BarConstructionResult, Tuple[BarConstructionResult, Dict[str, Any]]]:
    """
    Run bar construction with optimized temporary configurations.
    
    Args:
        tick_data: Input tick data
        base_config: Base configuration (uses default if None)
        optimization_mode: Optimization mode ("adaptive", "robust", "fast")
        return_optimization_details: Whether to return optimization details
        
    Returns:
        BarConstructionResult or tuple with optimization details
    """
    if base_config is None:
        base_config = BarConstructionConfig()
    
    config_manager, optimized_config = create_optimization_context(
        base_config, tick_data, optimization_mode
    )
    
    try:
        # Create constructor with optimized config
        constructor = EventBasedBarConstructor(optimized_config)
        
        # Run construction with temporary configurations
        result = constructor.construct_bars(tick_data)
        
        if return_optimization_details:
            optimization_summary = config_manager.get_optimization_summary()
            optimization_summary.update({
                'optimization_mode': optimization_mode,
                'convergence_achieved': config_manager.check_convergence(),
                'early_stopped': config_manager.should_early_stop(),
                'config_compatibility_warnings': validate_temporary_configuration_compatibility(
                    config_manager.optimization_config,
                    config_manager.validation_config,
                    config_manager.memory_config,
                    config_manager.performance_config
                )
            })
            return result, optimization_summary
        else:
            return result
            
    finally:
        config_manager.cleanup()


# Export all temporary configuration classes and functions
__all__ = [
    # Enums
    'BarTriggerType', 'OptimizationPhase', 'OptimizationStrategy',
    
    # Main configuration classes
    'BarConstructionConfig', 'BarConstructionResult', 'EventBasedBarConstructor',
    
    # Temporary configuration classes
    'TemporaryOptimizationConfig', 'TemporaryParameterConfig', 'TemporaryValidationConfig',
    'TemporaryMemoryConfig', 'TemporaryPerformanceConfig', 'TemporaryConfigurationManager',
    
    # Context managers
    'temporary_optimization_config', 'temporary_validation_config',
    'temporary_memory_config', 'temporary_performance_config',
    
    # Factory functions
    'create_temporary_optimization_config', 'create_temporary_parameter_config',
    'create_temporary_validation_config', 'create_temporary_memory_config',
    'create_temporary_performance_config', 'create_temporary_configuration_manager',
    
    # Utility functions
    'validate_temporary_configs', 'merge_temporary_configs',
    'create_optimization_parameter_space', 'create_parameter_constraints',
    'create_adaptive_temporary_configs', 'create_robust_temporary_configs',
    'create_fast_temporary_configs', 'validate_temporary_configuration_compatibility',
    'create_optimization_context', 'run_optimized_bar_construction',
    
    # Convenience functions
    'create_bar_construction_manager', 'construct_bars'
]